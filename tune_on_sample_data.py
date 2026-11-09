#!/usr/bin/env python3
"""Tune possession_pipeline.py on Metrica's OPEN sample games, never on the competition match.

Put this file next to possession_pipeline.py (repo root) and unzip
https://github.com/metrica-sports/sample-data into that folder, so this exists:

    sample-data-master/data/Sample_Game_1/...
    sample-data-master/data/Sample_Game_2/...
    sample-data-master/data/Sample_Game_3/...

Then:
    python tune_on_sample_data.py

What it does, per sample game
-----------------------------
1. Practice answers. Builds an H/A/D label for every frame from the game's event data: the team
   of the latest on-ball event (pass, shot, recovery, set piece, ball lost) has the ball; ball
   out, fouls and goals start a stoppage that lasts until the next restart.
2. Make it as hard as our match. The sample tracking is cleaner than ours, so it is degraded the
   same ways ours is degraded: the ball is hidden on about 41% of frames (more often in crowded
   moments), and each player's track is chopped into pieces that jump between anonymous slots.
3. Grid search. Runs our pipeline with different settings and scores each run with the official
   competition metric (evaluation/possession_metrics.py) against the practice answers.

Prints the best settings and writes tuning_results.csv and tuned_params.json to the folder
that contains this repo (change with --out-dir).
Deterministic: fixed random seed, same output every run.
"""

import argparse
import csv
import glob
import itertools
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_pipeline as pp  # noqa: E402
import possession_metrics as pm   # noqa: E402

PITCH = (105.0, 68.0)             # Metrica sample games: 105 x 68 m
FPS = 25
SEED = 2027
ON_BALL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}

# Our current settings expressed on a common scale (temperature 1): cost / 0.15.
BASE_TEMPERATURE = pp.PARAMS["temperature"]
GRID = dict(
    switch_eff=[8, 15, 25, 40],          # original default: 6.0 / 0.15 = 40, tuned: 15
    dead_eff=[15, 33],                   # current default: 5.0 / 0.15 = 33
    min_poss_s=[0.4, 1.0],               # current default: 1.0
    dead_speed_ms=[1.4, 2.0],            # current default: 1.4
)
DEFAULT = dict(switch_eff=15, dead_eff=33, min_poss_s=1.0, dead_speed_ms=1.4)


# --------------------------------------------------------------------------------------------
# Loading the sample games into one common shape
# --------------------------------------------------------------------------------------------

class Game(object):
    """players (N, P, 2) metres, team_of_player (P,), ball (N, 2) metres, events list."""

    def __init__(self, name, players, team_of_player, ball, events):
        self.name, self.players, self.team_of_player = name, players, team_of_player
        self.ball, self.events = ball, events


def _xy_columns(columns, prefix):
    """Positions of the x columns whose header starts with prefix (y is the next column)."""
    return [i for i, c in enumerate(columns) if str(c).startswith(prefix)]


def load_csv_game(folder):
    """Sample Games 1 and 2: two tracking CSVs (home, away) and an events CSV."""
    def find(pattern):
        hits = glob.glob(os.path.join(folder, pattern))
        if not hits:
            raise IOError("no file matching {} in {}".format(pattern, folder))
        return hits[0]

    def read_tracking(path):
        df = pd.read_csv(path, skiprows=2, low_memory=False)
        cols = list(df.columns)
        frames = df["Frame"].values.astype(np.int64)
        data = df.values
        pcols = _xy_columns(cols, "Player")
        players = np.stack([np.stack([data[:, i], data[:, i + 1]], 1).astype(np.float64)
                            for i in pcols], 1)
        bcols = _xy_columns(cols, "Ball")
        ball = (np.stack([data[:, bcols[0]], data[:, bcols[0] + 1]], 1).astype(np.float64)
                if bcols else None)
        return frames, players, ball

    f_home, p_home, b_home = read_tracking(find("*RawTrackingData_Home_Team.csv"))
    f_away, p_away, b_away = read_tracking(find("*RawTrackingData_Away_Team.csv"))
    n = int(max(f_home.max(), f_away.max()))
    players = np.full((n, p_home.shape[1] + p_away.shape[1], 2), np.nan)
    players[f_home - 1, :p_home.shape[1]] = p_home
    players[f_away - 1, p_home.shape[1]:] = p_away
    ball = np.full((n, 2), np.nan)
    ball_src = b_home if b_home is not None else b_away
    ball[f_home - 1] = ball_src
    team = np.array([0] * p_home.shape[1] + [1] * p_away.shape[1])

    ev = pd.read_csv(find("*RawEventsData.csv"))
    events = []
    for _, r in ev.iterrows():
        side = str(r["Team"]).strip().lower()
        events.append(dict(
            team=0 if side == "home" else 1 if side == "away" else None,
            type=str(r["Type"]).strip().upper(),
            sub=str(r.get("Subtype", "") or "").strip().upper(),
            period=int(r["Period"]),
            start=int(r["Start Frame"]) - 1,
            end=int(r["End Frame"]) - 1 if not pd.isna(r["End Frame"]) else int(r["Start Frame"]) - 1))
    return Game(os.path.basename(folder), _to_metres(players), team, _to_metres(ball), events)


def load_epts_game(folder):
    """Sample Game 3: EPTS metadata XML + tracking TXT + events JSON."""
    meta_path = glob.glob(os.path.join(folder, "*metadata.xml"))[0]
    trk_path = glob.glob(os.path.join(folder, "*tracking.txt"))[0]
    ev_path = glob.glob(os.path.join(folder, "*events.json"))[0]

    root = ET.parse(meta_path).getroot()
    teams = [t for t in root.iter("Team")]
    team_ids = [t.get("id") for t in teams]
    team_names = [(t.findtext("Name") or "").strip() for t in teams]
    player_ids = [p.get("id") for p in root.iter("Player")]
    player_team = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = []
        top = spec.find("SplitRegister")
        for group in (top.findall("SplitRegister") if top is not None else []):
            ref = group.find("PlayerChannelRef")
            if ref is not None:
                order.append(channel_player.get(ref.get("playerChannelId")))
        specs.append((int(spec.get("startFrame", 0)), int(spec.get("endFrame", 10 ** 9)), order))
    if not specs:
        raise ValueError("no DataFormatSpecification in {}".format(meta_path))

    col = {pid: i for i, pid in enumerate(player_ids)}
    rows = {}
    with open(trk_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) < 2 or not parts[0].strip():
                continue
            frame = int(parts[0])
            order = next((o for s, e, o in specs if s <= frame <= e), specs[0][2])
            xy = np.full((len(player_ids), 2), np.nan)
            for k, group in enumerate(parts[1].split(";")):
                if k >= len(order) or order[k] not in col:
                    continue
                vals = group.split(",")
                if len(vals) >= 2:
                    xy[col[order[k]]] = (_f(vals[0]), _f(vals[1]))
            b = (np.nan, np.nan)
            if len(parts) > 2:
                vals = parts[2].split(",")
                if len(vals) >= 2:
                    b = (_f(vals[0]), _f(vals[1]))
            rows[frame] = (xy, b)
    n = max(rows)
    players = np.full((n, len(player_ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for frame, (xy, b) in rows.items():
        players[frame - 1], ball[frame - 1] = xy, b
    team = np.array([player_team[pid] for pid in player_ids])

    with open(ev_path) as handle:
        raw = json.load(handle)
    raw = raw.get("data", raw) if isinstance(raw, dict) else raw
    events = []
    for e in raw:
        tm = e.get("team") or {}
        tid, tname = tm.get("id"), (tm.get("name") or "").strip()
        if tid in team_ids:
            side = team_ids.index(tid)
        elif tname in team_names:
            side = team_names.index(tname)
        else:
            side = None
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        sub = " ".join(str((s or {}).get("name", "")).upper() for s in (subs or []))
        start = (e.get("start") or {}).get("frame")
        end = (e.get("end") or {}).get("frame") or start
        if start is None:
            continue
        events.append(dict(team=side, type=str((e.get("type") or {}).get("name", "")).upper(),
                           sub=sub, period=int(e.get("period", 1)),
                           start=int(start) - 1, end=int(end) - 1))
    return Game(os.path.basename(folder), _to_metres(players), team, _to_metres(ball), events)


def _f(text):
    text = text.strip()
    try:
        return float(text) if text else np.nan
    except ValueError:
        return np.nan


def _to_metres(arr):
    out = np.array(arr, dtype=np.float64)
    out[..., 0] *= PITCH[0]
    out[..., 1] *= PITCH[1]
    return out


# --------------------------------------------------------------------------------------------
# Practice answers from events
# --------------------------------------------------------------------------------------------

def practice_labels(game):
    """Per-frame H/A/D/X codes (possession_metrics convention) built from the event data."""
    n = len(game.ball)
    codes = np.full(n, pm.X, np.int8)
    blocks = []
    events = sorted(game.events, key=lambda e: e["start"])
    for period in sorted({e["period"] for e in events}):
        evs = [e for e in events if e["period"] == period]
        start = max(0, min(e["start"] for e in evs))
        end = min(n - 1, max(max(e["end"], e["start"]) for e in evs))
        changes = [(start, pm.D)]
        for e in evs:
            side = e["team"]
            if e["type"] in STOPS:
                changes.append((e["start"], pm.D))
            elif e["type"] in ON_BALL and side is not None:
                changes.append((e["start"], pm.H if side == 0 else pm.A))
                if e["type"] == "SHOT" and "GOAL" in e["sub"]:
                    changes.append((max(e["end"], e["start"] + 1), pm.D))
        changes = [(min(max(f, start), end), c) for f, c in changes]
        changes.sort(key=lambda fc: fc[0])
        for i, (f, c) in enumerate(changes):
            stop = changes[i + 1][0] if i + 1 < len(changes) else end + 1
            codes[f:stop] = c
        blocks.append((len(blocks) + 1, start, end))
    return codes, blocks


# --------------------------------------------------------------------------------------------
# Degrade the clean sample tracking to look like the competition's automatic tracking
# --------------------------------------------------------------------------------------------

def degrade(game, graded, rng, target_missing=0.41, slots_per_team=24, mean_tracklet_s=40.0):
    n = len(game.ball)
    slots = np.full((n, 2 * slots_per_team, 2), np.nan, np.float32)
    slot_team = np.array([0] * slots_per_team + [1] * slots_per_team, np.int8)
    for team in (0, 1):
        free_at = np.zeros(slots_per_team, np.int64)
        pieces = []
        for j in np.where(game.team_of_player == team)[0]:
            seen = ~np.isnan(game.players[:, j, 0])
            for s, e in zip(*pp.runs_of(seen)):
                t = s
                while t < e:
                    length = int(rng.exponential(mean_tracklet_s * FPS)) + FPS
                    pieces.append((t, min(e, t + length), j))
                    t += length
        pieces.sort()
        for s, e, j in pieces:
            free = np.where(free_at <= s)[0]
            if not len(free):
                continue                              # tracker had no slot: the piece is lost
            k = int(rng.choice(free))
            slots[s:e, team * slots_per_team + k] = game.players[s:e, j]
            free_at[k] = e + int(rng.integers(1, 50))
    # short random dropouts of individual slots (about 3% of observations)
    drop = rng.random((n, slots.shape[1])) < 0.0015
    for _ in range(4):
        drop[1:] |= drop[:-1]
    slots[drop] = np.nan

    # ball: hide it in chunks, more often when the ball is in a crowd
    ball = game.ball + rng.normal(0, 0.2, game.ball.shape)
    dist = np.sqrt(((game.players - game.ball[:, None, :]) ** 2).sum(-1))
    crowd = (np.nan_to_num(dist, nan=99.0) < 3.0).sum(1) >= 3
    gap_mean = 1.2 * FPS
    lo, hi = 0.0, 0.2
    for _ in range(12):                               # binary search on the gap start rate
        base = (lo + hi) / 2
        hidden = _gaps(n, base, crowd, gap_mean, np.random.default_rng(SEED))
        missing = (hidden | np.isnan(ball[:, 0]))[graded].mean()
        lo, hi = (base, hi) if missing < target_missing else (lo, base)
    hidden = _gaps(n, (lo + hi) / 2, crowd, gap_mean, np.random.default_rng(SEED))
    ball[hidden] = np.nan
    return slots, slot_team, ball.astype(np.float32)


def _gaps(n, base, crowd, gap_mean, rng):
    hidden = np.zeros(n, bool)
    starts = rng.random(n) < base * np.where(crowd, 3.0, 1.0)
    lengths = (rng.exponential(gap_mean, n) + 1).astype(np.int64)
    t = 0
    for s in np.where(starts)[0]:
        if s < t:
            continue
        t = s + lengths[s]
        hidden[s:t] = True
    return hidden


def attack_sign(game, blocks):
    """+1 if the home team (team 0) attacks towards +x in the first period."""
    _, s, _ = blocks[0]
    home = game.players[s:s + 50, game.team_of_player == 0, 0]
    return 1.0 if np.nanmean(home) < PITCH[0] / 2 else -1.0


# --------------------------------------------------------------------------------------------
# Run the pipeline with one set of settings
# --------------------------------------------------------------------------------------------

def decode(E, aux, blocks, n, switch_eff, dead_eff, min_poss_s):
    P = dict(pp.PARAMS, switch_cost=switch_eff, dead_in_cost=dead_eff, dead_out_cost=dead_eff,
             min_poss_s=min_poss_s)
    pred = np.full(n, pm.X, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        path = pp.viterbi(E[seg], P)
        path = pp.merge_short(path, P)
        path = pp.snap_boundaries(path, {k: v[seg] for k, v in aux.items()}, P)
        pred[seg] = path
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample-dir", default=os.path.join(ROOT, "sample-data-master", "data"))
    ap.add_argument("--out-dir", default=os.path.dirname(ROOT),
                    help="where to write the results (default: the folder containing this repo)")
    args = ap.parse_args()
    out_dir = args.out_dir
    t0 = time.time()

    folders = sorted(glob.glob(os.path.join(args.sample_dir, "Sample_Game_*")))
    if not folders:
        sys.exit("No Sample_Game_* folders in {}. Unzip metrica-sports/sample-data there."
                 .format(args.sample_dir))

    prepared = []
    for folder in folders:
        try:
            game = (load_epts_game(folder) if glob.glob(os.path.join(folder, "*metadata.xml"))
                    else load_csv_game(folder))
        except Exception as error:
            print("SKIP {}: could not read ({}: {})".format(
                os.path.basename(folder), type(error).__name__, error))
            continue
        gt, blocks = practice_labels(game)
        stats = pm.possession_stats(gt, pm.valid_runs(gt))
        rng = np.random.default_rng(SEED)
        slots, slot_team, ball = degrade(game, gt != pm.X, rng)
        sign = attack_sign(game, blocks)
        print("{}: {} frames, practice answer has {} possessions of mean {:.1f} s, "
              "ball hidden on {:.0f}% of graded frames".format(
                  game.name, len(gt), stats["n_spells"], stats["mean_duration_s"],
                  100 * np.isnan(ball[gt != pm.X, 0]).mean()))
        prepared.append(dict(name=game.name, gt=gt, blocks=blocks, slots=slots,
                             slot_team=slot_team, ball=ball, sign=sign))
    if not prepared:
        sys.exit("No sample game could be read. Paste this output to get help.")

    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print("\ntrying {} settings on {} games ...".format(len(combos), len(prepared)))
    results = []
    evidence = {}
    for g in prepared:
        for ds in GRID["dead_speed_ms"]:
            P = dict(pp.PARAMS, dead_speed_ms=ds, temperature=1.0)
            evidence[(g["name"], ds)] = pp.build_evidence(
                g["slots"], g["ball"], g["slot_team"], g["sign"], g["blocks"], [],
                ("Home", "Away"), PITCH, P, lambda *a, **k: None)

    for combo in combos:
        c = dict(zip(keys, combo))
        row = dict(c)
        spell, macro = [], []
        for g in prepared:
            E, aux = evidence[(g["name"], c["dead_speed_ms"])]
            pred = decode(E, aux, g["blocks"], len(g["gt"]),
                          c["switch_eff"], c["dead_eff"], c["min_poss_s"])
            r = pm.evaluate(g["gt"], pred)
            spell.append(r["spell_f1_mean"])
            macro.append(r["macro_f1"])
            row["spellF1_" + g["name"]] = round(r["spell_f1_mean"], 4)
            row["poss_" + g["name"]] = "{}/{}".format(r["n_pred_spells"], r["n_gt_spells"])
        row["spellF1_mean"] = round(float(np.mean(spell)), 4)
        row["macroF1_mean"] = round(float(np.mean(macro)), 4)
        results.append(row)

    results.sort(key=lambda r: (-r["spellF1_mean"], -r["macroF1_mean"]))
    with open(os.path.join(out_dir, "tuning_results.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    def show(label, r):
        per = "  ".join("{} {:.3f} (poss {})".format(
            g["name"].replace("Sample_Game_", "G"), r["spellF1_" + g["name"]],
            r["poss_" + g["name"]]) for g in prepared)
        print("{:<9} switch {:>2}  dead {:>2}  min_poss {:.1f}s  dead_speed {:.1f}  |  "
              "spellF1 {:.4f}  macroF1 {:.4f}  |  {}".format(
                  label, r["switch_eff"], r["dead_eff"], r["min_poss_s"], r["dead_speed_ms"],
                  r["spellF1_mean"], r["macroF1_mean"], per))

    print("\nTop 5 settings (poss = predicted / practice-answer possessions):")
    for i, r in enumerate(results[:5]):
        show("#{}".format(i + 1), r)
    current = next(r for r in results if all(r[k] == DEFAULT[k] for k in keys))
    show("CURRENT", current)

    best = results[0]
    tuned = dict(switch_cost=round(best["switch_eff"] * BASE_TEMPERATURE, 3),
                 dead_in_cost=round(best["dead_eff"] * BASE_TEMPERATURE, 3),
                 dead_out_cost=round(best["dead_eff"] * BASE_TEMPERATURE, 3),
                 min_poss_s=best["min_poss_s"], dead_speed_ms=best["dead_speed_ms"])
    with open(os.path.join(out_dir, "tuned_params.json"), "w") as handle:
        json.dump(tuned, handle, indent=2)
    print("\nBest settings for possession_pipeline.py (temperature {}): {}".format(
        BASE_TEMPERATURE, tuned))
    print("Gain over current settings on the practice games: spellF1 {:+.4f}".format(
        best["spellF1_mean"] - current["spellF1_mean"]))
    print("Wrote tuning_results.csv and tuned_params.json to {}  ({:.0f} s)".format(out_dir, time.time() - t0))


if __name__ == "__main__":
    main()
