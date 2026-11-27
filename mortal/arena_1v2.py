import prelude

import argparse
import glob
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from os import path

import numpy as np
import torch
from engine import MortalEngine
from evaluate import load_checkpoint
from libriichi.arena import OneVsThree
from libriichi.consts import MAX_VERSION, NUM_PLAYERS
from libriichi.stat import Stat
from config import config, GAME_LENGTH

USAGE = '''Seat-rotation evaluation: one candidate against copies of a baseline.

Each seed is a fixed sequence of walls, played once per seat (three games in
sanma, four otherwise) with the candidate in a different seat each time and
the baseline in the others. A seed's score is the candidate's mean place over
those games, and seeds, not games, are the samples: with identical models the
games of a seed can be exact copies. Equal strength is a mean place of 2.0 in
sanma (2.5 in four-player); lower is better.

Games are played in chunks under OUT_DIR, so an interrupted run resumes where
it stopped. Both sides play greedily (no exploration), in fp32, with the
rule-based agari guard on, on the device in `[eval] device`. The summary goes
to OUT_DIR/summary.json.'''

CANDIDATE = 'candidate'
BASELINE = 'baseline'


def sha256(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_commit():
    here = path.dirname(path.abspath(__file__))
    try:
        commit = subprocess.check_output(['git', '-C', here, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', here, 'status', '--porcelain'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ('-dirty' if dirty else '')


def build_engine(filename, name, device):
    version, steps, (mortal, dqn, _) = load_checkpoint(filename, device)
    assert version == MAX_VERSION, f'{filename} is version {version}, this libriichi plays {MAX_VERSION}'
    engine = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = device,
        enable_amp = False,
        enable_rule_based_agari_guard = True,
        name = name,
    )
    return engine, steps


def play(args, device):
    candidate, candidate_steps = build_engine(args.candidate, CANDIDATE, device)
    baseline, baseline_steps = build_engine(args.baseline, BASELINE, device)
    logging.info(f'candidate: {args.candidate} ({candidate_steps:,} steps)')
    logging.info(f'baseline: {args.baseline} ({baseline_steps:,} steps)')

    end = args.seed_start + args.seeds
    for start in range(args.seed_start, end, args.chunk):
        count = min(args.chunk, end - start)
        chunk_dir = path.join(args.out_dir, f'chunk_{start}')
        if path.isdir(chunk_dir):
            logging.info(f'seeds {start}..{start + count - 1}: done before, skipping')
            continue
        tmp_dir = chunk_dir + '.tmp'
        if path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)  # left by an interrupted run
        t0 = time.time()
        env = OneVsThree(
            disable_progress_bar = True,
            log_dir = tmp_dir,
            game_length = GAME_LENGTH,
        )
        env.py_vs_py(
            challenger = candidate,
            champion = baseline,
            seed_start = (start, args.key),
            seed_count = count,
        )
        os.rename(tmp_dir, chunk_dir)
        secs = time.time() - t0
        games = count * NUM_PLAYERS
        logging.info(f'seeds {start}..{start + count - 1}: {games} games in {secs:.0f} s ({games / secs:.2f} games/s)')
    return candidate_steps, baseline_steps


def candidate_results(log_file):
    """The candidate's place (1 = top) and point change in one game log."""
    with gzip.open(log_file, 'rt', encoding='utf-8') as f:
        raw = f.read()
    names = json.loads(raw[:raw.index('\n')])['names']
    assert names.count(CANDIDATE) == 1, f'{log_file}: candidate seats {names}'
    stat = Stat.from_log(raw, names.index(CANDIDATE))
    ranks = [stat.rank_1, stat.rank_2, stat.rank_3, stat.rank_4][:NUM_PLAYERS]
    return ranks.index(1) + 1, stat.point


def per_seed(args):
    """Each seed's mean candidate place and points, and the games by place."""
    places, points = [], []
    games_by_place = [0] * NUM_PLAYERS
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        files = glob.glob(path.join(args.out_dir, 'chunk_*[0-9]', f'{seed}_{args.key}_*.json.gz'))
        assert len(files) == NUM_PLAYERS, f'seed {seed}: expected {NUM_PLAYERS} games, found {len(files)}'
        results = [candidate_results(f) for f in sorted(files)]
        places.append(np.mean([r[0] for r in results]))
        points.append(np.mean([r[1] for r in results]))
        for place, _ in results:
            games_by_place[place - 1] += 1
    return np.array(places), np.array(points), games_by_place


def bootstrap_ci(x, rng, resamples):
    means = np.empty(resamples)
    for i in range(0, resamples, 1000):
        idx = rng.integers(0, len(x), size=(min(1000, resamples - i), len(x)))
        means[i:i + len(idx)] = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def sign_flip_p(d, rng, resamples):
    """Two-sided p for mean(d) = 0, flipping each seed's sign at random.

    Under equal strength the candidate is just another seat, so its place in a
    seed is symmetric around the middle place and each deviation is equally
    likely with either sign.
    """
    observed = abs(d.mean())
    extreme = 0
    for i in range(0, resamples, 1000):
        signs = rng.choice((-1., 1.), size=(min(1000, resamples - i), len(d)))
        extreme += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).sum())
    return (extreme + 1) / (resamples + 1)


def summarize(args, places, points, games_by_place, steps):
    rng = np.random.default_rng(args.stats_seed)
    middle = (NUM_PLAYERS + 1) / 2
    return {
        'candidate': {'file': path.abspath(args.candidate), 'sha256': sha256(args.candidate), 'steps': steps[0]},
        'baseline': {'file': path.abspath(args.baseline), 'sha256': sha256(args.baseline), 'steps': steps[1]},
        'mortal_commit': git_commit(),
        'device': config['eval']['device'],
        'game_length': GAME_LENGTH,
        'num_players': NUM_PLAYERS,
        'seed_start': args.seed_start,
        'seeds': args.seeds,
        'key': args.key,
        'games': int(args.seeds * NUM_PLAYERS),
        'equal_strength_place': middle,
        'mean_place': float(places.mean()),
        'mean_place_ci95': bootstrap_ci(places, rng, args.resamples),
        'p_two_sided': sign_flip_p(places - middle, rng, args.resamples),
        'per_seed_place_sd': float(places.std(ddof=1)),
        'mean_points': float(points.mean()),
        'mean_points_ci95': bootstrap_ci(points, rng, args.resamples),
        'games_by_place': {str(i + 1): n for i, n in enumerate(games_by_place)},
        'stats': {
            'unit': 'seed (mean over its seat rotations)',
            'ci': 'percentile bootstrap over seeds',
            'p': 'sign-flip permutation test on seed place minus the middle place, plus-one corrected',
            'resamples': args.resamples,
            'stats_seed': args.stats_seed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out_dir', help='where the game logs and summary.json go')
    parser.add_argument('candidate', help='checkpoint to test')
    parser.add_argument('baseline', help='checkpoint it plays against, in every other seat')
    parser.add_argument('--seeds', type=int, default=1000, help='number of seeds (default 1000)')
    parser.add_argument('--seed-start', type=int, default=1_000_000, help='first seed (default 1000000)')
    parser.add_argument('--key', type=int, default=0x5EA7, help='seed key shared by all games (default 0x5EA7)')
    parser.add_argument('--chunk', type=int, default=100, help='seeds per chunk (default 100)')
    parser.add_argument('--resamples', type=int, default=100_000, help='bootstrap and permutation resamples')
    parser.add_argument('--stats-seed', type=int, default=20261126, help='seed for the resampling')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(config['eval']['device'])
    steps = play(args, device)
    places, points, games_by_place = per_seed(args)
    summary = summarize(args, places, points, games_by_place, steps)
    with open(path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    lo, hi = summary['mean_place_ci95']
    plo, phi = summary['mean_points_ci95']
    print(f'candidate {path.basename(args.candidate)} vs {NUM_PLAYERS - 1}x baseline {path.basename(args.baseline)}')
    print(f'seeds {args.seeds} ({summary["games"]} games), equal strength = {summary["equal_strength_place"]}')
    print(f'mean place {summary["mean_place"]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  p = {summary["p_two_sided"]:.4g}')
    print(f'per-seed place SD {summary["per_seed_place_sd"]:.4f}')
    print(f'mean points {summary["mean_points"]:+,.0f}  95% CI [{plo:+,.0f}, {phi:+,.0f}]')
    print(f'games by place {summary["games_by_place"]}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
