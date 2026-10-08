"""Valid spawn/goal scenarios for CARLA **Town03** only.

These indices and xyz points match the Town03 spawn layout. Do not point
``--map-name Town04`` (or any other map) at this file — a missing or
wrong table is a hard error, not a silent Town03 fallback.

To add another town, copy this file to ``<map>.py`` with the **lowercased
map name** as the module name (``Town04`` → ``town04.py``) and replace
spawn/goal indices for that map. Keep the same ``SCENARIOS`` dict shape
and scenario ids you want to use from the CLI. ``carla_env`` loads
``importlib.import_module(map_name.lower())``.

Each scenario is a dict:

- ``select``: ``cycle`` (use then increment) or ``random``
- ``routes``: list of ``(spawn_index, goal)`` where ``goal`` is a spawn
  index (int) or ``(x, y, z)``
- optional ``spawn_dy`` applied to the spawn location
- optional ``spawn_range`` (lo, hi) instead of a spawn index
- optional ``goal`` = ``random_other_spawn`` (scenario 12)
- optional ``middle_patches``: list of ``("set", i, xyz)`` /
  ``("insert", i, xyz)`` / ``("append_copy", i)``

Commented tuples inside lists are unused variants — keep them there.
"""

# Scenario 10: goal xyz; spawn is random in spawn_range.
_SC10_GOALS = [
    (50, 203.913498, 0.275307),
    (100, 203.788742, 1.3),
    (-55.387177, 0.558450, 0.0),
    (-105.387177, -3.140184, 0.0),
    (74, -40, 1.0),
    (-6.5, -44, 0.0),
]

# Scenario 11: (spawn index, goal xyz)
_SC11 = [
    (14, (-55.387177, 0.558450, 0.0)),
    (14, (-105.387177, -3.140184, 0.0)),
]

# Scenario 12: spawn-point indices (goal is a random other spawn)
_SC12 = (
    167, 165, 18, 200, 222, 105, 106, 1, 134, 3, 100, 139, 109, 190,
    146, 188, 186, 184, 174, 126, 48, 233, 0, 32, 30, 44, 191, 51, 53,
    238, 237, 235, 231, 229, 228, 225, 11, 85, 45, 247, 132, 252, 42,
)

# Scenario 13: (spawn index, goal spawn index) left
_SC13 = [
    (78, 76),
    (71, 130),
    (37, 161),
    (43, 222),
    (204, 162),
]

# Scenario 14: (spawn index, goal spawn index) right
_SC14 = [  # right
    # (28, 154), (49, 132), (83, 225), (77, 200), (54, 235),  # straight
    (28, 155),
    (49, 129),
    (83, 89),
    (77, 98),
    (54, 234),
]

# Scenario 15: (spawn index, goal spawn index) straight
_SC15 = [  # straight
    # (78, 76), (71, 130), (37, 161), (43, 222), (204, 162),  # left
    (78, 92),
    (71, 131),
    (238, 130),
    (43, 89),
    (204, 67),
]

# Scenario 16: testing pair
_SC16 = [
    (28, 154),
]

SCENARIOS = {
    1: {
        "select": "single",
        "routes": [(3, (50, 203.913498, 0.275307))],
    },
    2: {
        "select": "single",
        "routes": [(3, (100, 203.788742, 1.3))],
    },
    3: {
        "select": "single",
        "routes": [(11, (-55.387177, 0.558450, 0.0))],
        "spawn_dy": -11,
        "middle_patches": [
            ("set", 0, (-70.599335, 1.434147, 0.0)),
        ],
    },
    4: {
        "select": "single",
        "routes": [(12, (-105.387177, -3.140184, 0.0))],
        "spawn_dy": -15,
        "middle_patches": [
            ("set", 0, (-91.646820, -2.737971, 0.0)),
            ("set", 1, (-83.688499, 0.805027, 0.0)),
            ("append_copy", 2),
            ("set", 2, (-99.680237, -3.129901, 0.0)),
        ],
    },
    5: {
        "select": "single",
        "routes": [(13, (-55.387177, 0.558450, 0.0))],
        "spawn_dy": 10,
        "middle_patches": [
            ("insert", 1, (-70.599335, 1.434147, 0.0)),
        ],
    },
    6: {
        "select": "single",
        "routes": [(14, (-105.387177, -3.140184, 0.0))],
        "middle_patches": [
            ("set", 1, (-83.688499, 0.805027, 0.0)),
        ],
    },
    7: {
        # create_scenario used spawn 130; plan_the_route fell through to this xyz
        "select": "single",
        "routes": [(130, (-6.5, -44, 0.0))],
    },
    8: {
        "select": "single",
        "routes": [(130, (74, -40, 1.0))],
    },
    10: {
        "select": "random",
        "spawn_range": (0, 30),
        "goal_xyz_list": _SC10_GOALS,
    },
    11: {
        "select": "random",
        "routes": _SC11,
    },
    12: {
        "select": "random",
        "spawn_indices": _SC12,
        "goal": "random_other_spawn",
    },
    13: {
        "select": "random",
        "routes": _SC13,
    },
    14: {
        "select": "cycle",
        "routes": _SC14,
    },
    15: {
        "select": "cycle",
        "routes": _SC15,
    },
    16: {
        "select": "cycle",
        "routes": _SC16,
    },
}
