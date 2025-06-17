from dataclasses import dataclass


@dataclass()
class ACTIONS:
    # forward = 0
    # forward_left = 1
    # forward_right = 2
    # brake = 3
    # brake_left = 4
    # brake_right = 5

    # ACTION_CONTROL = {
    #     # acc, br, steer
    #     0: [0.5, 0, 0],  # forward
    #     1: [0.5, 0, -0.5],  # forward left
    #     2: [0.5, 0, 0.5],  # forward right
    #     3: [0, 1, 0],  # brake
    #     4: [0, 1, -0.5],  # brake left
    #     5: [0, 1, 0.5],  # brake right
    # }

    # ACTIONS_NAMES = {
    #     0: 'forward',
    #     1: 'forward_left',
    #     2: 'forward_right',
    #     3: 'brake',
    #     4: 'brake_left',
    #     5: 'brake_right',
    # }
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
        6: [0.5, 0, -0.2], # forward slight left
        7: [0.5, 0, 0.2], #forward slight right
        8: [0, 1, -0.2], #brake slight left
        9: [0, 1, 0.2], #brake slight right

    }

    # ACTION_CONTROL = {
    #     # acc, br, steer
    #     0: [1, 0, 0],  # forward
    #     1: [0, 0, 0],  # no action
    #     2: [0, 0, 0.5],  # sharp right right
    #     3: [0, 0, -0.5],  # sharp left
    #     4: [0, 0, 0.2],  # slight right
    #     5: [0, 0, -0.2],  # slight left
    #     6: [0, 0.5, 0], # brake
    #     7: [0, 0.2, 0.2], #brake slight right
    #     8: [0, 0.2, -0.2], #brake slight left
    #     9: [0.4, 0, 0], # slight forward

    # }

    ACTIONS_NAMES = {
        0: 'forward',
        1: 'forward_left',
        2: 'forward_right',
        3: 'brake',
        4: 'brake_left',
        5: 'brake_right',
        6: 'forward_slight_left',
        7: 'forward_slight_right',
        8: 'brake_slight_left',
        9: 'brake_slight_right',
    }

    ACTIONS_VALUES = {y: x for x, y in ACTIONS_NAMES.items()}
