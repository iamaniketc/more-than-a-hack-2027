#!/usr/bin/env python3
"""More than a Hack 2027, Phase 1: frame-level ball possession (H / A / D / X).

Drop this file in the ROOT of your clone of metrica-sports/more-than-a-hack-2027 (next to
games.py), put the match data in ./data/tactical/, then:

    python possession_pipeline.py                      # writes evaluation/results/<team>.csv
    python evaluation/validate_submission.py evaluation/results/<team>.csv
    python evaluation/run_evaluation.py                # sanity check on the public 5 minutes

numpy only. No labels from the provided game are used anywhere: the only "training" is a
team-shape model fitted on pseudo-labels that the code itself derives from confident
ball-at-feet frames (allowed under RULES.md: algorithmic pseudo-labels, no human decisions).

Pipeline
--------
1. Read tracking (players + ball) in metres. Fill short ball gaps (<= 0.5 s) by interpolation.
2. Per-frame evidence
   - ball control: distance from the ball to the nearest home vs nearest away player
   - team shape (for the 41% of frames with no ball): territory, spread, length, width,
     forward speed. Logistic model fitted on pseudo-labels.
   - dead ball: ball outside the pitch, players slowing down, few tracked players,
     long ball gaps, a stationary ball, Smart Tagging set pieces and goals
   - Smart Tagging phases as a weak team prior (attacking phases -> that team,
     defending phases -> the other team)
3. Segment with a 3-state Viterbi decoder (H, A, D) that charges a cost for every change, so
   the output is whole possessions instead of flickering frames (what spellF1 rewards).
4. Clean up: merge possessions/stoppages shorter than 1 s, then snap each change of
   possession to the frame where the new team's player actually reaches the ball.
5. Write X outside the two graded halves, H/A/D inside them.

All parameters live in PARAMS. Do NOT tune them on the public 5 minutes (RULES.md): tune on
open data from other matches, or leave the defaults.
"""

import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected

FPS = GAMES[GAME]["fps"]
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

PARAMS = dict(
    ball_gap_fill_s=0.5,      # interpolate ball gaps up to this long
    control_radius_m=1.5,     # "at the feet" distance, used for pseudo-labels and boundary snap
    ball_scale_m=1.5,         # softness of the home-vs-away distance comparison
    near_ball_m=3.0,          # ball this close to someone: strong evidence
    far_ball_m=8.0,           # ball this far from everyone (in flight): weak evidence
    w_ball_near=1.0,
    w_ball_far=0.2,
    w_shape=0.7,              # weight of the team-shape model
    w_tag=0.4,                # weight of Smart Tagging team priors
    w_tag_dead=1.0,           # weight of Smart Tagging dead-ball priors
    dead_speed_ms=1.4,        # mean player speed below which play looks stopped
    min_tracked=12,           # fewer tracked players than this: camera off the pitch
    temperature=0.15,         # frames are correlated, so each frame's evidence is scaled down
    switch_cost=2.25,         # Viterbi cost of H <-> A (tuned on Metrica open sample games, see tune_on_sample_data.py)
    dead_in_cost=5.0,         # cost of alive -> D
    dead_out_cost=5.0,        # cost of D -> alive
    min_poss_s=1.0,           # shorter possessions are merged into their neighbours
    min_dead_s=1.0,           # shorter stoppages are merged into their neighbours
    refine_window_s=1.5,      # how far a change of possession may be moved when snapping
)

POSSESSING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                    "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------

def read_meta(meta_path):
    """Slot -> team (0 home, 1 away), goalkeeper flags, team names, home attack sign in half 1."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    home_id = teams[0].get("id")
    names = tuple((t.findtext("Name") or "").strip() for t in teams[:2])
    home_dir = "left_to_right"
    for param in teams[0].iter("ProviderParameter"):
        if param.findtext("Name") == "attack_direction_first_half":
            home_dir = (param.findtext("Value") or home_dir).strip()
    players = list(root.iter("Player"))
    slot_team = np.array([0 if p.get("teamId") == home_id else 1 for p in players], np.int8)
    goalkeeper = np.array([any(pp.findtext("Name") == "position_type"
                               and pp.findtext("Value") == "Goalkeeper"
                               for pp in p.iter("ProviderParameter")) for p in players])
    sign_half1 = 1.0 if home_dir == "left_to_right" else -1.0
    return slot_team, goalkeeper, names, sign_half1


def _floats(text):
    try:
        return np.array(text.split(","), dtype=np.float64)
    except ValueError:  # empty fields
        return np.array([float(v) if v.strip() else np.nan for v in text.split(",")])


def read_atd(raw_path, n_frames, n_slots, pitch):
    """Players (N, slots, 2) and ball (N, 2) in metres, NaN where missing. Row = video frame."""
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # ATD frame_id counts from 1, video from 0
            if not 0 <= frame < n_frames:
                continue
            values = _floats(parts[1].replace(";", ","))
            if values.size == n_slots * 3:
                values = values.reshape(n_slots, 3)
                players[frame, :, 0] = values[:, 0] * length
                players[frame, :, 1] = values[:, 1] * width
            b = _floats(parts[2])
            if b.size >= 2:
                ball[frame] = (b[0] * length, b[1] * width)
    return players, ball


def read_tags(tag_path):
    """[(code, start_frame, end_frame, team_name)] from the SportsCode XML."""
    out = []
    if not tag_path or not os.path.isfile(tag_path):
        return out
    for inst in ET.parse(tag_path).iter("instance"):
        try:
            start, end = float(inst.findtext("start")), float(inst.findtext("end"))
        except (TypeError, ValueError):
            continue
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append(((inst.findtext("code") or "").strip().lower(),
                    int(round(start * FPS)), int(round(end * FPS)),
                    (tags.get("Team") or "").strip().lower(),
                    (tags.get("Type") or "").strip().lower()))
    return out


# --------------------------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------------------------

def runs_of(mask):
    """[start, end) of every run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def fill_short_gaps(values, max_gap):
    """Linear interpolation across NaN gaps of at most max_gap frames, bounded on both sides."""
    out = values.copy()
    good = ~np.isnan(values)
    if good.sum() < 2:
        return out
    idx = np.arange(len(values))
    interp = np.interp(idx, idx[good], values[good])
    for s, e in zip(*runs_of(~good)):
        if s > 0 and e < len(values) and e - s <= max_gap:
            out[s:e] = interp[s:e]
    return out


def moving_mean(values, half_width):
    """Centred NaN-aware moving average."""
    v = np.where(np.isnan(values), 0.0, values).astype(np.float64)
    c = (~np.isnan(values)).astype(np.float64)
    cv, cc = np.concatenate(([0], np.cumsum(v))), np.concatenate(([0], np.cumsum(c)))
    i = np.arange(len(values))
    lo, hi = np.maximum(0, i - half_width), np.minimum(len(values), i + half_width + 1)
    n = cc[hi] - cc[lo]
    return np.where(n > 0, (cv[hi] - cv[lo]) / np.maximum(n, 1), np.nan)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_logistic(X, y, l2=1.0, iters=50):
    """Newton-Raphson logistic regression with an L2 penalty. Returns (w, b)."""
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(k + 1)
    reg = np.eye(k + 1) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) + reg @ w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w[:-1], w[-1]


# --------------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------------

def build_evidence(players, ball_raw, slot_team, sign_half1, blocks, tags, names, pitch, P, log):
    length, width = pitch
    n = len(ball_raw)
    home, away = slot_team == 0, slot_team == 1

    # Attack direction per frame: +1 if home attacks towards x = length.
    sign = np.full(n, sign_half1)
    for half, first, last in blocks:
        if half == 2:
            sign[first:] = -sign_half1

    # Ball, with short gaps filled.
    gap = int(round(P["ball_gap_fill_s"] * FPS))
    ball = np.stack([fill_short_gaps(ball_raw[:, 0].astype(np.float64), gap),
                     fill_short_gaps(ball_raw[:, 1].astype(np.float64), gap)], 1)
    raw_seen = ~np.isnan(ball_raw[:, 0])
    seen = ~np.isnan(ball[:, 0])
    log("ball seen on {:.1f}% of frames raw, {:.1f}% after filling gaps <= {} frames".format(
        100 * raw_seen.mean(), 100 * seen.mean(), gap))

    # Distances ball -> nearest player of each team.
    dist = np.sqrt(((players - ball[:, None, :].astype(np.float32)) ** 2).sum(-1))
    d_home = np.fmin.reduce(dist[:, home], axis=1)
    d_away = np.fmin.reduce(dist[:, away], axis=1)
    del dist

    # Player speeds from positions (independent of the file's speed unit).
    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * FPS
    spd = np.sqrt((vel ** 2).sum(-1))
    spd[spd > 12.0] = np.nan                     # identity switches create fake jumps
    mean_speed = moving_mean(np.nanmean(spd, axis=1), FPS)
    n_tracked = moving_mean((~np.isnan(players[:, :, 0])).sum(1).astype(np.float64), FPS)

    # Ball speed (on filled track).
    bvel = np.full(n, np.nan)
    bvel[1:] = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(1)) * FPS
    ball_speed = moving_mean(bvel, FPS // 2)

    # ---- Team shape features (direction-adjusted so "forward" means towards the goal attacked)
    x_dir = (players[:, :, 0] - length / 2.0) * sign[:, None]   # + = towards away goal
    def team_stats(mask, s):
        xs, ys = x_dir[:, mask] * s, players[:, mask, 1]
        vx = vel[:, mask, 0] * sign[:, None] * s
        return (np.nanmean(xs, 1), np.nanstd(xs, 1) + np.nanstd(ys, 1),
                np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1),
                np.nanmean(np.clip(vx, -10, 10), 1))
    hc, hs, hl, hw, hv = team_stats(home & True, 1.0)
    ac, as_, al, aw, av = team_stats(away & True, -1.0)
    raw_feats = [np.nanmean(x_dir, 1),           # territory: + means play in away half
                 hc + ac,                        # both blocks' push (home fwd minus away fwd)
                 hs - as_, hl - al, hw - aw,     # spread, length, width differences
                 hv - av]                        # forward running difference
    F = np.stack([moving_mean(f, FPS) for f in raw_feats], 1)
    F_ok = ~np.isnan(F).any(1)

    # Pseudo-labels: ball clearly at one team's feet, play alive. Algorithmic, no human input.
    r = P["control_radius_m"]
    alive_like = mean_speed > P["dead_speed_ms"]
    ph_home = raw_seen & (d_home < r) & (d_away > 2 * r) & alive_like & F_ok
    ph_away = raw_seen & (d_away < r) & (d_home > 2 * r) & alive_like & F_ok
    idx = np.where(ph_home | ph_away)[0]
    p_shape = np.full(n, 0.5)
    if len(idx) > 500 and ph_home.sum() > 100 and ph_away.sum() > 100:
        mu, sd = F[idx].mean(0), F[idx].std(0) + 1e-6
        Z = (F[idx] - mu) / sd
        w, b = fit_logistic(Z, ph_home[idx].astype(float))
        acc = ((sigmoid(Z @ w + b) > 0.5) == ph_home[idx]).mean()
        Zall = (np.where(F_ok[:, None], F, mu) - mu) / sd
        p_shape = np.where(F_ok, sigmoid(Zall @ w + b), 0.5)
        log("shape model: {} pseudo-labelled frames (H {}, A {}), in-sample accuracy {:.1%}"
            .format(len(idx), int(ph_home.sum()), int(ph_away.sum()), acc))
    else:
        log("shape model skipped: not enough pseudo-labelled frames")

    # ---- Smart Tagging priors
    tag_team = np.zeros(n)
    tag_dead = np.zeros(n)
    home_name, away_name = (names[0].lower(), names[1].lower())
    def side(team):
        if not team or team in ("n/a", "na"):
            return 0.0
        if team == home_name or "barc" in team:
            return 1.0
        if team == away_name or "atl" in team or "athl" in team:
            return -1.0
        return 0.0
    for code, s, e, team, kind in tags:
        s, e = max(0, s), min(n, max(e, s + 1))
        t = side(team)
        if code in POSSESSING_CODES:
            tag_team[s:e] += t
        elif code in DEFENDING_CODES:
            tag_team[s:e] -= t
        elif code == "set pieces":
            cut = s + int(0.4 * (e - s))
            tag_dead[s:cut] += 1.0              # the stoppage before the restart
            tag_team[cut:e] += t                # the taking team has it afterwards
        elif code == "goals":
            tag_dead[e:min(n, e + 10 * FPS)] += 1.0
    tag_team = np.clip(tag_team, -1, 1)
    tag_dead = np.clip(tag_dead, 0, 1)

    # ---- Dead-ball probability
    margin = 0.5
    out = seen & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin) |
                  (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    out = moving_mean(out.astype(float), 3) > 0.5
    slow = sigmoid((P["dead_speed_ms"] - np.nan_to_num(mean_speed, nan=0.0)) / 0.3)
    few = np.nan_to_num(n_tracked, nan=0.0) < P["min_tracked"]
    long_missing = np.zeros(n, bool)
    for s, e in zip(*runs_of(~seen)):
        if e - s > 3 * FPS:
            long_missing[s:e] = True
    static = seen & (np.nan_to_num(ball_speed, nan=9.0) < 0.3)
    dead_logit = (-1.5 + 4.0 * out + 3.5 * (2 * slow - 1) + 4.0 * few + 1.0 * long_missing
                  + 0.8 * static + P["w_tag_dead"] * tag_dead)
    p_dead = sigmoid(dead_logit)

    # ---- Team probability
    dmin = np.fmin(d_home, d_away)
    have_ball = seen & ~np.isnan(dmin)
    l_ball = np.clip(np.nan_to_num((d_away - d_home) / P["ball_scale_m"]), -4, 4)
    span = P["far_ball_m"] - P["near_ball_m"]
    w_ball = P["w_ball_far"] + (P["w_ball_near"] - P["w_ball_far"]) * np.clip(
        (P["far_ball_m"] - np.nan_to_num(dmin, nan=99.0)) / span, 0, 1)
    w_ball = np.where(have_ball, w_ball, 0.0)
    l_shape = np.clip(np.log(p_shape / (1 - p_shape)), -3, 3)
    team_logit = w_ball * l_ball + P["w_shape"] * l_shape + P["w_tag"] * 2.0 * tag_team
    p_home = sigmoid(team_logit)

    eps = 1e-4
    pd = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    E = np.stack([np.log(1 - pd) + np.log(ph), np.log(1 - pd) + np.log(1 - ph), np.log(pd)], 1)
    aux = dict(d_home=d_home, d_away=d_away, out=out, seen=seen, raw_seen=raw_seen)
    return E * P["temperature"], aux


# --------------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------------

def viterbi(E, P):
    cost = np.array([[0.0, P["switch_cost"], P["dead_in_cost"]],
                     [P["switch_cost"], 0.0, P["dead_in_cost"]],
                     [P["dead_out_cost"], P["dead_out_cost"], 0.0]])
    T = len(E)
    back = np.zeros((T, 3), np.int8)
    score = E[0].copy()
    cols = np.arange(3)
    for t in range(1, T):
        cand = score[:, None] - cost
        back[t] = cand.argmax(0)
        score = cand[back[t], cols] + E[t]
    path = np.empty(T, np.int8)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def segments(path):
    change = np.concatenate(([0], np.where(np.diff(path) != 0)[0] + 1, [len(path)]))
    return [(int(change[i]), int(change[i + 1]), int(path[change[i]]))
            for i in range(len(change) - 1)]


def merge_short(path, P):
    """Absorb possessions / stoppages shorter than the minimum into the surrounding state."""
    min_len = {H: int(P["min_poss_s"] * FPS), A: int(P["min_poss_s"] * FPS),
               D: int(P["min_dead_s"] * FPS)}
    path = path.copy()
    for _ in range(5):
        segs = segments(path)
        changed = False
        for i, (s, e, lab) in enumerate(segs):
            if e - s >= min_len[lab] or len(segs) == 1:
                continue
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                path[s:e] = prev
                changed = True
            elif lab != D and prev == D and nxt == D:
                continue                          # brief restart then out again: keep
            elif prev is not None and nxt is not None and prev != D and nxt != D:
                # short A inside H -> A handover etc.: give it to the following possession
                path[s:e] = nxt
                changed = True
        if not changed:
            break
    return path


def snap_boundaries(path, aux, P):
    """Move each H<->A change to the first frame where the new team actually reaches the ball,
    and each alive->D change to the first frame the ball is out of the pitch."""
    path = path.copy()
    w = int(P["refine_window_s"] * FPS)
    r = P["control_radius_m"]
    segs = segments(path)
    for i in range(1, len(segs)):
        s_prev, _, lab_prev = segs[i - 1]
        s, e, lab = segs[i]
        lo, hi = max(s_prev + 1, s - w), min(e - 1, s + w)
        if hi <= lo:
            continue
        if lab_prev != D and lab != D:
            d_new = aux["d_home"] if lab == H else aux["d_away"]
            d_old = aux["d_away"] if lab == H else aux["d_home"]
            ok = aux["seen"][lo:hi] & (d_new[lo:hi] < r) & ~(d_old[lo:hi] < d_new[lo:hi])
            hits = np.where(ok)[0]
            if len(hits):
                t = lo + int(hits[0])
                path[min(t, s):max(t, s)] = lab if t < s else lab_prev
        elif lab_prev != D and lab == D:
            hits = np.where(aux["out"][lo:hi])[0]
            if len(hits):
                t = lo + int(hits[0])
                path[min(t, s):max(t, s)] = D if t < s else lab_prev
    return path


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def run(out_path, data_dir=None, params=None, verbose=True, copy_dir=None):
    P = dict(PARAMS, **(params or {}))
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    try:
        tag_path = require(["smart_tagging"], GAME, data_dir)[0]
    except IOError:
        tag_path = None
        log("Smart Tagging not found: running without it")
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    blocks = half_video_blocks(GAME)

    slot_team, _, names, sign_half1 = read_meta(meta_path)
    log("H = {}, A = {}, {} track slots, home attacks {} in half 1".format(
        names[0], names[1], len(slot_team), "+x" if sign_half1 > 0 else "-x"))
    log("reading tracking ...")
    players, ball = read_atd(raw_path, n_frames, len(slot_team), pitch)
    tags = read_tags(tag_path)
    log("  done in {:.0f} s; {} Smart Tagging instances".format(time.time() - t0, len(tags)))

    E, aux = build_evidence(players, ball, slot_team, sign_half1, blocks, tags, names, pitch,
                            P, log)
    del players

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        seg = slice(first, last + 1)
        path = viterbi(E[seg], P)
        path = merge_short(path, P)
        sub_aux = {k: v[seg] for k, v in aux.items()}
        path = snap_boundaries(path, sub_aux, P)
        labels[seg] = LETTERS[path]
        segs = segments(path)
        poss = [s for s in segs if s[2] != D]
        alive = sum(e - s for s, e, _ in poss)
        log("half {}: {} possessions (H {}, A {}), mean {:.1f} s, {} stoppages, D {:.1f}%".format(
            half, len(poss), sum(1 for s in poss if s[2] == H),
            sum(1 for s in poss if s[2] == A), alive / FPS / max(len(poss), 1),
            sum(1 for s in segs if s[2] == D), 100.0 * np.mean(path == D)))

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    log("wrote {} rows -> {}  ({:.0f} s total)".format(len(labels), out_path, time.time() - t0))
    if copy_dir and os.path.isdir(copy_dir):
        import shutil
        copy_path = os.path.join(copy_dir, os.path.basename(out_path))
        shutil.copyfile(out_path, copy_path)
        log("copy  -> {}".format(copy_path))
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="my_team", help="team name, used for the file name")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--data-dir", default=None, help="defaults to ./data or MTAH_DATA_DIR")
    ap.add_argument("--copy-to", default=os.path.dirname(ROOT),
                    help="also copy the CSV here (default: the folder containing this repo); "
                         "pass '' to skip")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}.csv".format(args.team))
    run(out, args.data_dir, copy_dir=args.copy_to or None)
