"""Experiment / environment settings for the A3C CARLA template.

Machine paths, container images, and secrets live in `.env`.
Algorithm hyperparameters live on the `train_a3c.py` CLI.
The agent interface (observations, actions) and the env reward live in
`rl_configuration.py`.
"""

import os


def load_dotenv(path=None):
    """Read KEY=VALUE from `.env` next to this file. Existing env vars win."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.isfile(path):
        return path
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return path


load_dotenv()

# Both pipelines run the CARLA servers on the machine that trains.
CARLA_HOST = 'localhost'
PORT = int(os.environ.get('CARLA_START_PORT', '2000'))
# One server uses three ports: RPC, streaming (RPC+1), secondary (RPC+2).
PORT_STEP = int(os.environ.get('CARLA_PORT_STEP', '5'))

MAP_NAME = os.environ.get('CARLA_MAP', 'Town03')
CAMERA_TYPE = 'semantic'
RES = 250
SCENARIO = [14]
ACTION_REPEAT = 2  # world ticks per agent decision
EPISODE_MAX_DECISIONS = 200  # episode length limit; 0 turns it off
SHOW_CAM = False
DRAW = False
