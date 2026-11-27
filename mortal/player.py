import torch
import numpy as np
import os
import shutil
import random
import secrets
import logging
from glob import glob
from os import path
from model import Brain, DQN
from engine import MortalEngine
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from libriichi.consts import NUM_PLAYERS
from config import config, GAME_LENGTH

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            name = 'mortal',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
            game_length = GAME_LENGTH,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

def load_baseline(state_file, device, enable_compile, enable_amp):
    state = torch.load(state_file, weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']
    stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    stable_dqn = DQN(version=version).eval()
    stable_mortal.load_state_dict(state['mortal'])
    stable_dqn.load_state_dict(state['current_dqn'])
    if enable_compile:
        stable_mortal.compile()
        stable_dqn.compile()

    return MortalEngine(
        stable_mortal,
        stable_dqn,
        is_oracle = False,
        version = version,
        device = device,
        enable_amp = enable_amp,
        enable_rule_based_agari_guard = True,
        name = 'baseline',
    )

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        self.baseline_device = torch.device(baseline_cfg['device'])
        self.baseline_compile = baseline_cfg['enable_compile']
        # for both sides; half precision measured no faster on an M4
        self.enable_amp = baseline_cfg.get('enable_amp', True)

        # Opponent pool: each session plays one opponent in every other seat,
        # `state_file` (the anchor) with probability `anchor_ratio`, otherwise
        # a uniform pick among the files matching `pool_glob`, such as the
        # trainee's own past checkpoints. Without a pool it is always the anchor.
        self.anchor_file = baseline_cfg['state_file']
        self.pool_glob = baseline_cfg.get('pool_glob')
        self.anchor_ratio = baseline_cfg.get('anchor_ratio', 1.0)
        self.baseline_file = None
        self.baseline_engine = None

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        # each seed is played once per seat
        self.seed_count = cfg['games'] // NUM_PLAYERS
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

    def pick_baseline(self):
        """Choose this session's opponent and load it if it changed."""
        pool = sorted(glob(self.pool_glob)) if self.pool_glob else []
        if not pool or random.random() < self.anchor_ratio:
            chosen = self.anchor_file
        else:
            chosen = random.choice(pool)
        if chosen != self.baseline_file:
            self.baseline_engine = load_baseline(chosen, self.baseline_device, self.baseline_compile, self.enable_amp)
            self.baseline_file = chosen
        return chosen

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        self.pick_baseline()
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = self.enable_amp,
            name = 'trainee',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
            game_length = GAME_LENGTH,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
