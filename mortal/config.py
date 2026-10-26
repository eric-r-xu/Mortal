import toml
import os
from libriichi.consts import HANCHAN

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)

# 8 for hanchan, 4 for East-only (tonpuu); 6 and 3 in sanma.
GAME_LENGTH = config.get('env', {}).get('game_length', HANCHAN)
