"""Train the possession classifier on OTHER games: IDSSE (7 Bundesliga matches) and Metrica's open sample games.

The provided match is never read here. Tracking is degraded to look like the provided ATD (ball hidden
on about 41% of frames, player tracks chopped across anonymous slots) by tune_on_sample_data.degrade.

IDSSE (Bassek et al. 2025, CC-BY 4.0): https://huggingface.co/datasets/pysport/idsse-data
Metrica sample data:                  https://github.com/metrica-sports/sample-data

    python train_learned_model.py --idsse-dir ../idsse-data --sample-dir ../sample-data/data

Labels: IDSSE frame labels come from the DFL's ball status and possession fields, Metrica's from its
event data (tune_on_sample_data.practice_labels). Both are folded so a spell shorter than
MIN_CONTROL_S bracketed by the other team's possession is a touch without control, not a change.
Deterministic: fixed seeds, same model every run.
"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import learned_possession as lp  # noqa: E402
import possession_metrics as pm  # noqa: E402
import tune_on_sample_data as tune  # noqa: E402

IDSSE_MATCHES = ("J03WOH", "J03WMX", "J03WN1", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
MIN_CONTROL_S = 2.0      # CHALLENGE.md: a touch without control is not a change of possession
TRAIN_STEP = 8           # every 8th graded frame; neighbouring frames are near-duplicates
PITCH_HALF = (52.5, 34.0)  # IDSSE coordinates are centred on the pitch
FPS = 25
HYPERPARAMETERS = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100,
                       l2_regularization=1.0, random_state=0)


class Game(object):
    def __init__(self, players, team_of_player, ball):
        self.players, self.team_of_player, self.ball = players, team_of_player, ball


def fold_short(codes, min_seconds):
    """Give alive spells shorter than min_seconds to the previous possession when both neighbours are alive."""
    out, minimum = codes.copy(), int(min_seconds * FPS)
    for _ in range(10):
        changed = False
        for start, stop in pm.valid_runs(out):
            segs = pm.segments(out, [(start, stop)])
            for i in range(1, len(segs) - 1):
                s, e, code = segs[i]
                before, after = segs[i - 1][2], segs[i + 1][2]
                if code != pm.D and e - s < minimum and before != pm.D and after != pm.D:
                    out[s:e] = before
                    changed = True
        if not changed:
            break
    return out


def load_idsse(idsse_dir, match):
    """players (N, P, 2) and ball (N, 2) in metres, H/A/D codes, and the two graded halves."""
    info = glob.glob(os.path.join(idsse_dir, "DFL_02_01_matchinformation*%s.xml" % match))[0]
    positions = glob.glob(os.path.join(idsse_dir, "DFL_04_03_positions_raw_observed*%s.xml" % match))[0]
    roles = {t.get("TeamId"): t.get("Role") for t in ET.parse(info).getroot().iter("Team")}
    sets = {"firstHalf": [], "secondHalf": []}
    ball = {}
    current = None
    for event, el in ET.iterparse(positions, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            current = dict(section=el.get("GameSection"), team=el.get("TeamId"),
                           person=el.get("PersonId"), n=[], xy=[], possession=[], status=[])
        elif event == "end" and el.tag == "Frame" and current is not None:
            current["n"].append(int(el.get("N")))
            current["xy"].append((float(el.get("X")), float(el.get("Y"))))
            if current["team"] == "BALL":
                current["possession"].append(int(el.get("BallPossession")))
                current["status"].append(int(el.get("BallStatus")))
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            if current["team"] == "BALL":
                ball[current["section"]] = current
            elif current["team"] in roles:             # skips referees
                sets[current["section"]].append(current)
            current = None
            el.clear()

    persons = sorted({(c["person"], c["team"]) for s in sets.values() for c in s})
    index = {person: i for i, (person, _) in enumerate(persons)}
    team_of_player = np.array([0 if roles[t] == "home" else 1 for _, t in persons], np.int8)
    sections = ("firstHalf", "secondHalf")
    total = sum(len(ball[s]["n"]) for s in sections)
    players = np.full((total, len(persons), 2), np.nan, np.float32)
    ball_xy = np.full((total, 2), np.nan, np.float32)
    codes = np.empty(total, np.int8)
    blocks, offset = [], 0
    for half, section in enumerate(sections, 1):
        b = ball[section]
        length, first_n = len(b["n"]), b["n"][0]
        ball_xy[offset:offset + length] = np.array(b["xy"])
        codes[offset:offset + length] = np.where(np.array(b["status"]) == 0, pm.D,
                                                 np.array(b["possession"]) - 1)
        for c in sets[section]:
            rows = np.array(c["n"]) - first_n
            ok = (rows >= 0) & (rows < length)
            players[offset + rows[ok], index[c["person"]]] = np.array(c["xy"])[ok]
        blocks.append((half, offset, offset + length - 1))
        offset += length
    for array in (players, ball_xy):
        array[..., 0] += PITCH_HALF[0]
        array[..., 1] += PITCH_HALF[1]
    return players, ball_xy, team_of_player, codes, blocks


def degraded(name, players, ball, team_of_player, codes, blocks):
    """One training game: labels, degraded tracking and attack direction."""
    source = Game(players, team_of_player, ball)
    rng = np.random.default_rng(tune.SEED)
    graded = np.zeros(len(codes), bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    slots, slot_team, ball_seen = tune.degrade(source, graded, rng)
    return dict(name=name, codes=codes, blocks=blocks, slots=slots, slot_team=slot_team,
                ball=ball_seen, sign=tune.attack_sign(source, blocks), graded=graded)


def training_rows(game):
    """Feature rows for a game and its team-swapped mirror, so home and away are symmetric."""
    codes = fold_short(game["codes"], MIN_CONTROL_S)
    idx = np.where(game["graded"])[0][::TRAIN_STEP]
    labels = codes[idx]
    mirrored = np.where(labels == pm.H, pm.A, np.where(labels == pm.A, pm.H, labels))
    X, names, _ = lp.compute_features(game["slots"], game["ball"], game["slot_team"],
                                      game["sign"], game["blocks"])
    Xm, _, _ = lp.compute_features(game["slots"], game["ball"], 1 - game["slot_team"],
                                   -game["sign"], game["blocks"])
    return np.vstack([X[idx], Xm[idx]]), np.concatenate([labels, mirrored]), names


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--idsse-dir", default=os.path.join(os.path.dirname(ROOT), "idsse-data"))
    ap.add_argument("--sample-dir", default=os.path.join(os.path.dirname(ROOT), "sample-data", "data"))
    ap.add_argument("--out", default=lp.MODEL_PATH)
    args = ap.parse_args()

    games = []
    for match in IDSSE_MATCHES:
        print("reading", match, flush=True)
        games.append(degraded(match, *load_idsse(args.idsse_dir, match)))
    for folder in sorted(glob.glob(os.path.join(args.sample_dir, "Sample_Game_*"))):
        game = (tune.load_epts_game(folder) if glob.glob(os.path.join(folder, "*metadata.xml"))
                else tune.load_csv_game(folder))
        codes, blocks = tune.practice_labels(game)
        print("reading", game.name, flush=True)
        games.append(degraded(game.name, game.players, game.ball, game.team_of_player,
                              codes, blocks))

    parts = [training_rows(g) for g in games]
    X, y = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    print("training on {} rows from {} games".format(len(y), len(games)), flush=True)
    clf = HistGradientBoostingClassifier(**HYPERPARAMETERS).fit(X, y)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(dict(clf=clf, names=parts[0][2], games=[g["name"] for g in games]), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
