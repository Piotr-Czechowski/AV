"""Agent interface (observations and actions) and the env reward function.

This file is the one description of what the agent sees and does:

- ``observation_spec(config)`` gives the shape and dtype of every
  observation key,
- ``ACTIONS`` is the discrete action table.

The network, the wrapper, the vehicle control, and the logs read from here.
"""

import math

REWARD_FROM_TP = 0
REWARD_FROM_MP = 0
REWARD_FROM_COL = 0
REWARD_FROM_INV = 0

# The episode ends when the vehicle is this far from the planned route.
OFFROUTE_THRESHOLD_M = 10.0

# One row per discrete action: (name, throttle, brake, steer). The row index
# is the action id the policy outputs.
ACTIONS = (
    ('forward', 0.5, 0.0, 0.0),
    ('forward_left', 0.5, 0.0, -0.5),
    ('forward_right', 0.5, 0.0, 0.5),
    ('brake', 0.0, 1.0, 0.0),
    ('brake_left', 0.0, 1.0, -0.5),
    ('brake_right', 0.0, 1.0, 0.5),
    ('forward_slight_left', 0.5, 0.0, -0.2),
    ('forward_slight_right', 0.5, 0.0, 0.2),
    ('brake_slight_left', 0.0, 1.0, -0.2),
    ('brake_slight_right', 0.0, 1.0, 0.2),
)
N_ACTIONS = len(ACTIONS)

IMAGE_CHANNELS = 3  # both cameras deliver three colour channels
NUM_MANEUVERS = 3  # 0 = left, 1 = straight, 2 = right


def observation_spec(config):
    """Shape and dtype of every observation key, without a batch dimension.

    This is a pure function of the config: the main process builds the
    network from it before any CARLA connection exists. The wrapper checks
    every observation against it.
    """
    return {
        'image': {'shape': (IMAGE_CHANNELS, config.res, config.res),
                  'dtype': 'float32'},
        'speed': {'shape': (1,), 'dtype': 'float32'},
        'maneuver': {'shape': (), 'dtype': 'int64',
                     'num_classes': NUM_MANEUVERS},
    }


def check_observation(obs, obs_spec):
    """Raise ValueError when ``obs`` does not match ``obs_spec``."""
    if set(obs) != set(obs_spec):
        raise ValueError(
            'observation keys {} do not match the spec keys {}'.format(
                sorted(obs), sorted(obs_spec)))
    for key, spec in obs_spec.items():
        value = obs[key]
        if tuple(value.shape) != tuple(spec['shape']) or \
                str(value.dtype) != spec['dtype']:
            raise ValueError(
                "observation '{}' is {} {}, the spec says {} {}".format(
                    key, value.dtype, tuple(value.shape),
                    spec['dtype'], tuple(spec['shape'])))
        num_classes = spec.get('num_classes')
        if num_classes is not None and \
                not (0 <= value.min() and value.max() < num_classes):
            raise ValueError(
                "observation '{}' has a value outside 0..{}".format(
                    key, num_classes - 1))


def reward_function(
    collision_history_list,
    invasion_counter,
    speed,
    route_distance,
    mp_static_reward,
    terminal_state_reward,
    on_junction,
    prev_speed,
):
    """Scalar reward and done flag computed by CarlaEnv."""
    if len(collision_history_list) != 0 or \
            route_distance >= OFFROUTE_THRESHOLD_M:
        done = True
        col_reward = REWARD_FROM_COL
    else:
        done = False
        col_reward = 0

    inv_reward = invasion_counter * REWARD_FROM_INV

    # Peak near 20 km/h.
    speed_reward = -1.2 + 4 * math.sin(speed / 10)
    if route_distance < 1.5:
        route_distance_reward = 1
        if on_junction and speed_reward > 0:
            route_distance_reward = route_distance_reward * 4
    else:
        route_distance_reward = -4 * math.sin(speed / 10)

    reward = (
        terminal_state_reward
        + col_reward
        + speed_reward
        + route_distance_reward
        + inv_reward
        + mp_static_reward
    )

    return reward, done
