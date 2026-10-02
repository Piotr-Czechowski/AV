# Extending the template

The A3C core (`a3c_core.py`, `run_a3c.py`) does not know what an observation contains or which network trains. You change the agent in three files and leave the core alone: `rl_configuration.py` (observations and actions), `carla_wrapper.py` (builds each observation), and `model.py` (the network).

## The contract

One observation is a dict of NumPy values without a batch dimension:

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `image` | float32 | `[C, H, W]` | camera frame in `[0, 1]` |
| `speed` | float32 | `[1]` | km/h divided by `SPEED_SCALE` (100) |
| `maneuver` | int64 | scalar | next turn: 0 = left, 1 = straight, 2 = right |

- `rl_configuration.observation_spec(config)` gives the shape and dtype of every key. The wrapper checks every observation against it.
- `a3c_core.obs_to_tensors(obs, device)` adds a batch dimension of 1 to every key and moves it to the worker's device.
- The model takes that dict, `forward(obs)`, and returns `(logits [1, n_actions], value [1, 1])`.

`ModelContractTests` in `tests/test_template_smoke.py` checks these shapes.

## A new network

Add the class to `model.py` and return it from `build_model`. It is a `torch.nn.Module` with the constructor `(obs_spec, n_actions, device)` and `forward(obs)` as in the contract. Read the input sizes from `obs_spec`, not from literals.

`--resume` and `--init-from` work only with a checkpoint from the same network.

## A new input

Example: a second camera, a frame stack, or a vector of waypoints.

1. `observation_spec`: add the key with its shape and dtype.
2. `CarlaA3CWrapper._observation`: add the NumPy value under that key.
3. The model's `forward`: read `obs['<key>']`.

`a3c_core.py` needs no change.

## Another action table

Edit `ACTIONS` in `rl_configuration.py`. Each row is `(name, throttle, brake, steer)` and the row index is the action id. The number of actions and the size of the policy head follow the table.

- `CarlaEnv.reset()` holds action `3` (`brake`) while a new episode settles. Keep a braking action in row 3 or change that line.
- Continuous actions are not supported: the policy is a categorical distribution over the rows.

## Another camera or resolution

- `--camera rgb|semantic` and `--res N` (square image) are CLI flags.
- A non-square image: change the `image` shape in `observation_spec`. The camera gets its height and width from the spec, and the default network accepts another image size without a change.
- Camera mount and field of view: `CAMERA_X`, `CAMERA_Z`, `CAMERA_PITCH`, `CAMERA_FOV` at the top of `carla_env.py`.
- A new sensor type: add `add_<name>_camera` and `process_<name>_img` to `carla_env.py`, extend the `camera_type` branches in `reset()` and `step()`, and add the name to the `--camera` choices in `train_a3c.py`. Set `IMAGE_CHANNELS` in `rl_configuration.py` when the channel count changes.

## A new map

1. Copy `town03.py` to `<map>.py` with the lowercased map name (`Town04` becomes `town04.py`).
2. Replace the spawn and goal indices. Keep the shape of the `SCENARIOS` dict (the docstring of `town03.py` describes it).
3. Run with `--map-name Town04 --scenario <id>`, or set `CARLA_MAP` in `.env`.

A missing file or scenario id is an error. Scenario ids 1 and 2 run without middle goals on turns (`CarlaEnv._draw_optimal_route_lines` skips them), so use other ids for new scenarios.
