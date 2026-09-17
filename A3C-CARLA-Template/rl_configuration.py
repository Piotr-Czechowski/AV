"""RL configuration for the A3C CARLA agent.

Holds the discrete action space definition and the reward function.
"""

import math
from dataclasses import dataclass

import run_settings


@dataclass()
class Actions:
    """
    Example implementation of discrete action space for RL agent
    """

    forward = 0
    forward_left = 1
    forward_right = 2
    brake = 3
    brake_left = 4
    brake_right = 5
    forward_slight_left = 6
    forward_slight_right = 7
    brake_slight_left = 8
    brake_slight_right = 9

    ACTION_CONTROL = {
        # acc, br, steer
        0: [0.5, 0, 0],  # forward
        1: [0.5, 0, -0.5],  # no
        2: [0.5, 0, 0.5],  # forward right
        3: [0, 1, 0],  # brake
        4: [0, 1, -0.5],  # brake left
        5: [0, 1, 0.5],  # brake right
        6: [0.5, 0, -0.2],  # forward slight left
        7: [0.5, 0, 0.2],  # forward slight right
        8: [0, 1, -0.2],  # brake slight left
        9: [0, 1, 0.2],  # brake slight right
    }

    ACTIONS_NAMES = {
        0: "forward",
        1: "forward_left",
        2: "forward_right",
        3: "brake",
        4: "brake_left",
        5: "brake_right",
        6: "forward_slight_left",
        7: "forward_slight_right",
        8: "brake_slight_left",
        9: "brake_slight_right",
    }

    ACTIONS_VALUES = {y: x for x, y in ACTIONS_NAMES.items()}


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

    if len(collision_history_list) != 0 or route_distance >= 10:
        done = True
        col_reward = run_settings.REWARD_FROM_COL
    else:
        done = False
        col_reward = 0

    inv_reward = invasion_counter * run_settings.REWARD_FROM_INV

    speed_reward = -1.2 + 4 * math.sin(speed / 10)  # pik jest w okolicach 20 km/h
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
