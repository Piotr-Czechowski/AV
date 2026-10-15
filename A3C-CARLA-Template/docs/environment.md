# Environment

`CarlaEnv` (`carla_env.py`) talks to CARLA. `CarlaA3CWrapper` (`carla_wrapper.py`) turns it into `reset()` and `step()` for the worker. `rl_configuration.py` defines the observation, the actions, and the legacy reward.

## Simulation

- **One worker, one world.** The client connects to `localhost` on the worker's port, checks that client and server versions match, and loads the map.
- **Synchronous mode.** The world advances only when the worker ticks it. One tick is 0.1 s of simulated time (`FIXED_DELTA_SECONDS`).
- **Ego vehicle.** A Tesla Model 3. It drives alone: no traffic and no pedestrians are spawned. `spawn_npc_vehicle` and `spawn_single_pedestrian` exist in `carla_env.py`, and nothing calls them.
- **Sensors.** One camera (`semantic` or `rgb`), a collision sensor, and a lane-invasion sensor. The camera sits 0.3 m forward and 2.5 m up, pitched 10° down, with a 75° field of view.

## Episode

`reset()`:

1. Destroy the actors of the previous episode. Every `--world-reload-interval` episodes the whole world is reloaded instead (`0` means never).
2. Pick a scenario id at random from `--scenario`, then a route from that scenario's table.
3. Plan the route from the spawn point to the goal. The plan gives the list of turns and the middle goals.
4. Spawn the vehicle and the sensors, hold the brake for 16 ticks, and return the first observation.

`step(action)`:

1. Apply the action and tick the world, `--action-repeat` times. With the default of 2, one decision is 0.2 s of simulated time.
2. Measure speed, distance to the goal, distance to the route, collisions, and lane invasions.
3. Compute the reward and the done flag.
4. Take the newest camera frame and build the observation.

An episode ends when one of these is true:

| Condition | Defined in |
|---|---|
| Any collision | `reward_function` in `rl_configuration.py` |
| The vehicle is 10 m or more from the route | `OFFROUTE_THRESHOLD_M` in `rl_configuration.py` |
| The vehicle is within 3 m of the goal. The episode counts as a success | `GOAL_RADIUS_M` in `carla_env.py` |
| `--episode-max-decisions` decisions were taken (default 200, which is 40 s; `0` turns it off) | `carla_wrapper.py` |

## Observation

A dict of NumPy values without a batch dimension. `observation_spec(config)` gives the shape and dtype of every key, and the wrapper checks every observation against it.

| Key | dtype | Shape | Content |
|---|---|---|---|
| `image` | float32 | `[3, res, res]` | Camera frame scaled to `[0, 1]`, channels in CARLA's BGR order. The semantic camera is drawn in the CityScapes colour palette |
| `speed` | float32 | `[1]` | km/h divided by `SPEED_SCALE` (100) |
| `maneuver` | int64 | scalar | Turn to take at the next junction: 0 = left, 1 = straight, 2 = right |

`maneuver` follows the planned route. It starts with the first turn of the plan and moves to the next one each time the vehicle leaves a junction. After the last turn it is 1.

## Actions

Ten discrete actions from the `ACTIONS` table in `rl_configuration.py`. Each one is half throttle or full brake, combined with one of five steering values. The numbers in the table are the action ids.

| | steer 0 | steer −0.5 | steer +0.5 | steer −0.2 | steer +0.2 |
|---|---|---|---|---|---|
| throttle 0.5 | 0 | 1 | 2 | 6 | 7 |
| brake 1.0 | 3 | 4 | 5 | 8 | 9 |

## Reward

`--reward-mode` selects one of two reward functions. The done flag is the same in both.

### `legacy` (default)

`reward_function` in `rl_configuration.py`, with the speed `v` in km/h:

```
speed term = -1.2 + 4 * sin(v / 10)
route term = +1                   closer than 1.5 m to the route
           = -4 * sin(v / 10)     otherwise: cancels the speed gain, the sum is -1.2
reward     = speed term + route term + event terms
```

- The speed term peaks near 16 km/h (+2.8). It is negative below about 3 km/h and above about 28 km/h.
- The event terms are `REWARD_FROM_MP` (a middle goal is reached), `REWARD_FROM_TP` (the goal is reached), `REWARD_FROM_COL` (the step that ends the episode by collision or by leaving the route), and `REWARD_FROM_INV` (a lane invasion). All four are 0 by default.
- `reward_function` can multiply the route term by 4 inside a junction. The wrapper does not pass the junction flag, so this multiplier is inactive.

### `shaped`

`_shape_reward` in `carla_wrapper.py`. The coefficients are the `DEFAULT_REWARD_*` constants in `train_a3c.py` and have no flag.

| Component | Value per decision | Default |
|---|---|---|
| `progress` | coefficient × metres gained towards the goal, limited to ±5 m | 1.0 |
| `target_speed` | coefficient × `(1 - min(2, abs(v - target) / target))` | 1.0, target 20 km/h |
| `route_penalty` | −coefficient × distance to the route, limited to the off-route threshold | 0.1 |
| `time_penalty` | constant | −0.01 |
| `goal_bonus` | once, when the goal is reached | +50 |
| `collision_penalty` | once, on a new collision | −50 |
| `offroute_penalty` | when the distance to the route reaches the threshold (10 m) | −25 |
| `lane_invasion_penalty` | on a step with a lane invasion | −5 |

The sum is clipped to ±`DEFAULT_REWARD_CLIP` (50). Every component is logged separately (`docs/logging.md`).

## Routes and scenarios

A scenario is a set of routes. The tables live in one file per map, named after the lowercased map name: `town03.py` for `Town03`. A missing file or a missing scenario id is an error. The environment never falls back to another map's table.

Each entry of `SCENARIOS` is a dict:

| Key | Meaning |
|---|---|
| `routes` | List of `(spawn index, goal)`. The goal is a spawn index or an `(x, y, z)` point |
| `select` | How a route is picked: `cycle` (in order), `random`, or `single` (always the first) |
| `spawn_dy` | Optional shift of the spawn point along y, in metres |
| `spawn_range`, `goal_xyz_list` | Instead of `routes`: random spawn index in a range, random goal from a list |
| `spawn_indices` with `goal: "random_other_spawn"` | Instead of `routes`: random spawn from a list, any other spawn point as the goal |
| `middle_patches` | Optional manual corrections of the middle goals |

Town03 scenarios:

| Id | Routes |
|---|---|
| 1 to 8 | One fixed route each |
| 10 | Random spawn among indices 0 to 30, random goal among six points |
| 11 | Two routes, random |
| 12 | Random spawn from a list, any other spawn point as the goal |
| 13 | Five left turns, random |
| 14 | Five right turns, in order (default) |
| 15 | Five straight routes, in order |
| 16 | One test route |

With several ids, `--scenario 14 15`, every episode first picks one id at random. The `cycle` position is kept per worker.

**Route.** `GlobalRoutePlanner` (`carla_navigation/`) traces the route as waypoints 1 m apart. The distance to the route is the distance to the nearest of these waypoints.

**Middle goals.** Points along the route: the entry and the exit of every turn, extra points that split a segment longer than 25 m (`DEFAULT_MP_DENSITY` in `train_a3c.py`), and the goal as the last one. The legacy reward pays `REWARD_FROM_MP` once for each. Reaching the last one ends the episode.

## Debugging aids

| Tool | Effect |
|---|---|
| `--save-episodes 10 500` | Saves every camera frame of these global episodes as JPEG under `episodes/<episode>-<port>/` |
| `--save-episode-interval N` | The same for every N-th global episode |
| `--verbose-env-logs` | Prints the planned maneuvers and spawn attempts |
| `DRAW` in `settings.py` | Draws the route in the simulator |
| `SHOW_CAM` in `settings.py` | Opens a window with the camera image |

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--map-name` | `Town03`, or `CARLA_MAP` | Map to load |
| `--scenario` | `14` | One or more scenario ids |
| `--camera` | `semantic` | Or `rgb` |
| `--res` | 250 | Width and height of the image in pixels |
| `--action-repeat` | 2 | World ticks per decision |
| `--episode-max-decisions` | 200 | Episode length limit |
| `--world-reload-interval` | 0 | Episodes of a worker between full world reloads |
| `--reward-mode` | `legacy` | Or `shaped` |

The simulation step, the client and camera timeouts, the ego vehicle, the camera mount, and the goal radius are constants at the top of `carla_env.py`.
