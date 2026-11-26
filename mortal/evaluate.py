import prelude

import sys
import json
import torch
import logging
from glob import glob
from torch import nn
from torch.utils.data import DataLoader
from model import Brain, DQN, AuxNet
from dataloader import FileDatasetsIter, worker_init_fn
from common import tqdm
from libriichi.consts import ACTION_SPACE, NUM_PLAYERS
from config import config

USAGE = '''Usage: python evaluate.py [--json OUT] CHECKPOINT [CHECKPOINT ...]

Scores checkpoints on held-out games (`[eval] globs`), reading the games once for
all of them. For each checkpoint it reports the training losses and how often its
top action matches the one taken in the log, overall and by kind of action.'''

# Action kinds by label; `kita` exists only in sanma.
ACTION_KINDS = {
    'discard': range(0, 37),
    'riichi': [37],
    'chi': [38, 39, 40],
    'pon': [41],
    'kan': [42],
    'agari': [43],
    'ryukyoku': [44],
    'pass': [45],
    'kita': [46],
}

def load_checkpoint(filename, device):
    state = torch.load(filename, weights_only=True, map_location=device)
    cfg = state['config']
    version = cfg['control']['version']
    mortal = Brain(version=version, **cfg['resnet']).to(device).eval()
    dqn = DQN(version=version).to(device).eval()
    aux_net = AuxNet((NUM_PLAYERS,)).to(device).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    aux_net.load_state_dict(state['aux_net'])
    return version, state.get('steps', 0), (mortal, dqn, aux_net)

def main():
    args = sys.argv[1:]
    json_file = None
    if len(args) >= 2 and args[0] == '--json':
        json_file = args[1]
        args = args[2:]
    if not args or any(a.startswith('-') for a in args):
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    cfg = config['eval']
    device = torch.device(cfg['device'])
    gamma = config['env']['gamma']
    batch_size = cfg['batch_size']

    checkpoints = []
    for filename in args:
        version, steps, models = load_checkpoint(filename, device)
        checkpoints.append((filename, version, steps, models))
    versions = {c[1] for c in checkpoints}
    assert len(versions) == 1, f'checkpoints mix obs versions {versions}'
    version = versions.pop()

    file_list = []
    for pat in cfg['globs']:
        file_list.extend(glob(pat, recursive=True))
    file_list.sort()
    assert file_list, 'no games match `[eval] globs`'
    logging.info(f'version: {version}')
    logging.info(f'games: {len(file_list):,}')

    file_data = FileDatasetsIter(
        version = version,
        file_list = file_list,
        pts = config['env']['pts'],
        file_batch_size = cfg.get('file_batch_size', 20),
    )
    data_loader = DataLoader(
        dataset = file_data,
        batch_size = batch_size,
        num_workers = cfg.get('num_workers', 1),
        worker_init_fn = worker_init_fn,
    )

    kinds = {k: v for k, v in ACTION_KINDS.items() if max(v) < ACTION_SPACE}
    label_kind = torch.zeros(ACTION_SPACE, dtype=torch.int64)
    for i, labels in enumerate(kinds.values()):
        label_kind[list(labels)] = i
    label_kind = label_kind.to(device)

    mse = nn.MSELoss(reduction='sum')
    ce = nn.CrossEntropyLoss(reduction='sum')
    sums = [{
        'dqn_loss': 0.,
        'cql_loss': 0.,
        'next_rank_loss': 0.,
        'agree': 0,
        'agree_by_kind': torch.zeros(len(kinds), dtype=torch.int64, device=device),
    } for _ in checkpoints]
    count = 0
    count_by_kind = torch.zeros(len(kinds), dtype=torch.int64, device=device)

    with torch.inference_mode():
        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in tqdm(data_loader, desc='EVAL'):
            bs = obs.shape[0]
            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            q_target_mc = (gamma ** steps_to_done * kyoku_rewards).to(dtype=torch.float32, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)

            kind = label_kind[actions]
            count += bs
            count_by_kind += torch.bincount(kind, minlength=len(kinds))

            for s, (_, _, _, (mortal, dqn, aux_net)) in zip(sums, checkpoints):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(bs), actions]
                s['dqn_loss'] += 0.5 * mse(q, q_target_mc).item()
                s['cql_loss'] += (q_out.logsumexp(-1) - q).sum().item()
                next_rank_logits, = aux_net(phi)
                s['next_rank_loss'] += ce(next_rank_logits, player_ranks).item()

                agree = q_out.argmax(-1) == actions
                s['agree'] += agree.sum().item()
                s['agree_by_kind'] += torch.bincount(kind[agree], minlength=len(kinds))

    results = []
    for s, (filename, _, steps, _) in zip(sums, checkpoints):
        by_kind = {
            k: s['agree_by_kind'][i].item() / n
            for i, k in enumerate(kinds)
            if (n := count_by_kind[i].item()) > 0
        }
        results.append({
            'checkpoint': filename,
            'steps': steps,
            'samples': count,
            'dqn_loss': s['dqn_loss'] / count,
            'cql_loss': s['cql_loss'] / count,
            'next_rank_loss': s['next_rank_loss'] / count,
            'agreement': s['agree'] / count,
            'agreement_by_kind': by_kind,
        })

    shares = ', '.join(
        f'{k} {count_by_kind[i].item() / count:.1%}'
        for i, k in enumerate(kinds)
        if count_by_kind[i].item() > 0
    )
    print(f'samples: {count:,} ({shares})')
    for r in results:
        print(f'\n{r["checkpoint"]} (steps {r["steps"]:,})')
        print(f'  dqn loss {r["dqn_loss"]:.4f}, cql loss {r["cql_loss"]:.4f}, next rank loss {r["next_rank_loss"]:.4f}')
        print(f'  agreement {r["agreement"]:.2%}: ' + ', '.join(
            f'{k} {v:.2%}' for k, v in r['agreement_by_kind'].items()
        ))
    if json_file is not None:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
