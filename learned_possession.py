"""Learned frame-level possession (H / A / D) from tracking, for the More than a Hack 2027 match.

A gradient-boosted classifier turns football features of each frame (who is near the ball, who last
controlled it, team shape, dead-ball cues, all with time context) into H / A / D probabilities. A
Viterbi decoder then picks whole possessions. The model is trained on other games only, see
train_learned_model.py. No labels from the provided match are used.

    python learned_possession.py --team "My Team"     # writes evaluation/results/<team>.csv
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import possession_pipeline as pp  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0
CONTROL_M = 1.5          # ball this close to a player counts as under control
CONTEXT_CONTROL_M = 3.0  # looser radius for the windowed "ball is with a player" feature
FILL_FRAMES = 12         # interpolate ball gaps up to 0.5 s, as possession_pipeline does
CAP = 250                # 10 s cap on frames-since / frames-until features
CONTEXT_WINDOWS = (25, 100, 250)   # centred moving-average half-widths, frames
CONTEXT_SHIFTS = (-50, -25, 25, 50)
# Decoder costs in log-probability units, picked leave-one-game-out on 7 IDSSE and 3 Metrica games
# (2026-11-08); the optimum is flat between switch 8-12 and dead 20-40.
SWITCH_COST, DEAD_COST = 12.0, 40.0
MODEL_PATH = os.path.join(ROOT, "models", "possession_hgb.joblib")
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def _since(mask):
    """(index of the last True at or before each frame, frames since it), capped."""
    n = len(mask)
    last = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
    return last, np.minimum(np.where(last >= 0, np.arange(n) - last, CAP), CAP)


def _carry_back(values, mask):
    """Value at the last mask frame at or before each frame (0 before the first), and frames since."""
    last, since = _since(mask)
    return np.where(last >= 0, values[np.maximum(last, 0)], 0.0), since


def _carry_forward(values, mask):
    """Value at the next mask frame at or after each frame (0 after the last), and frames until."""
    out, until = _carry_back(values[::-1], mask[::-1])
    return out[::-1], until[::-1]


def compute_features(slots, ball_raw, slot_team, sign_half1, blocks):
    """Per-frame feature matrix, its column names, and what the boundary logic needs.

    Args:
        slots: (N, slots, 2) player positions in metres, NaN where missing.
        ball_raw: (N, 2) ball position in metres, NaN where missing.
        slot_team: (slots,) 0 for the home team, 1 for away.
        sign_half1: +1 if home attacks towards x = pitch length in the first half.
        blocks: [(half, first_frame, last_frame)] graded halves.
    """
    n = len(ball_raw)
    home, away = slot_team == 0, slot_team == 1
    sign = np.full(n, sign_half1)
    for half, first, _ in blocks:
        if half == 2:
            sign[first:] = -sign_half1

    ball = np.stack([pp.fill_short_gaps(ball_raw[:, i].astype(np.float64), FILL_FRAMES)
                     for i in (0, 1)], 1)
    raw_seen = ~np.isnan(ball_raw[:, 0])
    seen = ~np.isnan(ball[:, 0])
    _, since_seen = _since(raw_seen)
    _, until_seen = _since(raw_seen[::-1])
    until_seen = until_seen[::-1]

    dist = np.sqrt(((slots - ball[:, None, :].astype(np.float32)) ** 2).sum(-1))
    d_home_all = np.sort(np.where(np.isnan(dist[:, home]), 99.0, dist[:, home]), 1)
    d_away_all = np.sort(np.where(np.isnan(dist[:, away]), 99.0, dist[:, away]), 1)
    d_home, d_away = d_home_all[:, 0], d_away_all[:, 0]
    d_min = np.minimum(d_home, d_away)

    def within(mask, radius):
        return (np.nan_to_num(dist[:, mask], nan=99.0) < radius).sum(1)

    vel = np.full(slots.shape, np.nan, np.float32)
    vel[1:] = (slots[1:] - slots[:-1]) * FPS
    spd = np.sqrt((vel ** 2).sum(-1))
    spd[spd > 12.0] = np.nan                       # identity switches create fake jumps
    mean_speed = pp.moving_mean(np.nanmean(spd, axis=1), FPS)
    n_tracked = pp.moving_mean((~np.isnan(slots[:, :, 0])).sum(1).astype(np.float64), FPS)

    ball_vel = np.full((n, 2), np.nan)
    ball_vel[1:] = (ball[1:] - ball[:-1]) * FPS
    ball_spd = np.sqrt((ball_vel ** 2).sum(1))
    ball_speed = pp.moving_mean(ball_spd, FPS // 4)
    ball_accel = pp.moving_mean(np.abs(np.diff(ball_spd, prepend=np.nan)), 3)
    ball_speed_max = np.fmax.reduce([np.roll(ball_speed, k) for k in (-25, -12, 0, 12, 25)])

    nearest = np.nanargmin(np.where(np.isnan(dist), 1e9, dist), axis=1)
    rows = np.arange(n)
    rel_speed = pp.moving_mean(np.sqrt(((vel[rows, nearest] - ball_vel.astype(np.float32)) ** 2)
                                       .sum(-1)).astype(np.float64), 3)
    nearest_speed = np.nan_to_num(spd[rows, nearest], nan=0.0)

    control = seen & (d_min < CONTROL_M)
    owner = np.where(d_home <= d_away, 1.0, -1.0)
    last_team, last_since = _carry_back(owner, control)
    next_team, next_until = _carry_forward(owner, control)

    # Team shape, with + meaning towards the away goal.
    x_dir = (slots[:, :, 0] - PITCH_L / 2.0) * sign[:, None]

    def shape_stats(mask, direction):
        xs, ys = x_dir[:, mask] * direction, slots[:, mask, 1]
        vx = vel[:, mask, 0] * sign[:, None] * direction
        return (np.nanmean(xs, 1), np.nanstd(xs, 1) + np.nanstd(ys, 1),
                np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1),
                np.nanmean(np.clip(vx, -10, 10), 1))

    hc, hs, hl, hw, hv = shape_stats(home, 1.0)
    ac, as_, al, aw, av = shape_stats(away, -1.0)
    shape = [pp.moving_mean(f, FPS) for f in
             (np.nanmean(x_dir, 1), hc, ac, hs - as_, hl - al, hw - aw, hv - av)]
    territory_long = pp.moving_mean(shape[0], 5 * FPS)
    run_long = pp.moving_mean(shape[-1], 5 * FPS)

    bx, by = ball[:, 0], ball[:, 1]
    outside = seen & ((bx < -0.5) | (bx > PITCH_L + 0.5) | (by < -0.5) | (by > PITCH_W + 0.5))
    outside = pp.moving_mean(outside.astype(float), 3)
    to_line = np.minimum.reduce([bx, PITCH_L - bx, by, PITCH_W - by])
    static = pp.moving_mean(((ball_speed < 0.3) & seen).astype(float), FPS)
    n3h, n3a = within(home, 3), within(away, 3)

    cols = dict(
        raw_seen=raw_seen.astype(float), since_seen=since_seen, until_seen=until_seen,
        d_home=np.where(seen, d_home, np.nan), d_away=np.where(seen, d_away, np.nan),
        d_diff=np.where(seen, d_away - d_home, np.nan), d_min=np.where(seen, d_min, np.nan),
        dh2=np.where(seen, d_home_all[:, 1], np.nan), da2=np.where(seen, d_away_all[:, 1], np.nan),
        n3h=n3h, n3a=n3a, n6h=within(home, 6), n6a=within(away, 6),
        ball_speed=ball_speed, ball_accel=ball_accel, ball_speed_max=ball_speed_max,
        rel=np.where(seen, rel_speed, np.nan), nearest_speed=nearest_speed,
        last_team=last_team, last_since=last_since, next_team=next_team, next_until=next_until,
        territory=shape[0], home_push=shape[1], away_push=shape[2], d_spread=shape[3],
        d_length=shape[4], d_width=shape[5], d_fwd_run=shape[6],
        terr_long=territory_long, run_long=run_long,
        mean_speed=mean_speed, n_tracked=n_tracked, out=outside,
        to_line=np.where(seen, to_line, np.nan), static=static,
        ball_fwd=np.where(seen, (bx - PITCH_L / 2.0) * sign, np.nan))

    context = {"d_diff": cols["d_diff"], "cnt_diff": n3h - n3a, "last_team": last_team,
               "fwd_run": cols["d_fwd_run"], "mean_speed": mean_speed, "ball_speed": ball_speed,
               "out": outside, "static": static, "territory": cols["territory"],
               "ctrl": np.where(seen, (d_min < CONTEXT_CONTROL_M).astype(float), np.nan)}
    for name, values in context.items():
        values = np.asarray(values, dtype=np.float64)
        for half_width in CONTEXT_WINDOWS:
            cols["%s_w%d" % (name, half_width)] = pp.moving_mean(values, half_width)
        for shift in CONTEXT_SHIFTS:
            cols["%s_s%d" % (name, shift)] = np.roll(values, shift)

    names = list(cols)
    X = np.stack([np.asarray(cols[k], dtype=np.float32) for k in names], 1)
    aux = dict(d_home=d_home, d_away=d_away, seen=seen)
    return X, names, aux


def decode(probs, blocks, switch_cost=SWITCH_COST, dead_cost=DEAD_COST):
    """Most likely H/A/D path per graded half from per-frame probabilities, X elsewhere."""
    params = dict(pp.PARAMS, switch_cost=switch_cost, dead_in_cost=dead_cost,
                  dead_out_cost=dead_cost)
    evidence = np.log(np.clip(probs, 1e-4, 1.0))
    labels = np.full(len(probs), "X", dtype="<U1")
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        path = pp.merge_short(pp.viterbi(evidence[seg], params), params)
        labels[seg] = LETTERS[path]
    return labels


def run(out_path, data_dir=None, model_path=MODEL_PATH):
    """Write the possession file for the provided match."""
    import joblib

    model = joblib.load(model_path)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    slot_team, _, _, sign_half1 = pp.read_meta(meta_path)
    players, ball = pp.read_atd(raw_path, source_frames("tactical", GAME), len(slot_team),
                                GAMES[GAME]["pitch"])
    blocks = half_video_blocks(GAME)
    X, names, _ = compute_features(players, ball, slot_team, sign_half1, blocks)
    if names != model["names"]:
        raise ValueError("feature columns differ from the trained model")
    labels = decode(model["clf"].predict_proba(X), blocks)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="my_team", help="team name, used for the file name")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--data-dir", default=None, help="defaults to ./data or MTAH_DATA_DIR")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}.csv".format(args.team))
    run(out, args.data_dir, args.model)
    print("wrote", out)
