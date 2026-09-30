import toml
import os

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)

# 8 for hanchan, 4 for East-only (tonpuu).
GAME_LENGTH = config.get('env', {}).get('game_length', 8)
