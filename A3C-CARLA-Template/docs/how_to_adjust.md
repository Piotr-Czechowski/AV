# How to adjust the template

How to adjust the template to your own agent: which files to edit, and in which order, to add or replace a network, an input, the actions, the reward, a config value, the camera, or the map. To change a value that already exists, see `docs/configuration.md`.

The A3C core (`a3c_core.py`, `run_a3c.py`) does not know what an observation contains or which network trains. You adjust the agent in three files and leave the core alone: `rl_configuration.py` (observations and actions), `carla_wrapper.py` (builds each observation and the shaped reward), and `model.py` (the network).

## The contract

- `rl_configuration.observation_spec(config)` gives the shape and dtype of every observation key. The default keys are described in `docs/environment.md`.
- The wrapper returns each observation as a dict of NumPy values without a batch dimension and checks it against the spec.
- `a3c_core.obs_to_tensors(obs, device)` adds a batch dimension of 1 to every key and moves it to the worker's device.
- The model takes that dict, `forward(obs)`, and returns `(logits [1, n_actions], value [1, 1])`.

`ModelContractTests` in `tests/test_template_smoke.py` checks these shapes.

## A new network

Add the class to `model.py` and return it from `build_model`. It is a `torch.nn.Module` with the constructor `(obs_spec, n_actions, device)` and `forward(obs)` as in the contract. Read the input sizes from `obs_spec`, not from literals.

- `--resume` and `--init-from` work only with a checkpoint from the same network.
- The worker calls the model once per step and keeps no hidden state. A recurrent network needs changes in the worker loop of `a3c_core.py`.

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

## Another reward

- **Legacy mode.** Edit `reward_function` and the `REWARD_FROM_*` constants in `rl_configuration.py`.
- **Shaped mode.** Add a term to the `components` dict in `CarlaA3CWrapper._shape_reward`. Every key of that dict is summed into the reward and logged under its own name.
- **Done flag.** Collision and off-route come from `reward_function`, the goal from `CarlaEnv.static_reward_mp`, the length limit from `CarlaA3CWrapper.step`. Both reward modes use the same flag.

## A new config value

The core and the wrapper read one config namespace, and every field is required.

1. Value that changes between runs: add a flag to `build_parser` in `train_a3c.py`. Constant: add a `DEFAULT_*` constant and a line in `_apply_config_defaults`.
2. Read it as `config.<name>` in `a3c_core.py` or `carla_wrapper.py`.

## Another camera or resolution

- `--camera rgb|semantic` and `--res N` (square image) are CLI flags.
- A non-square image: change the `image` shape in `observation_spec`. The camera gets its height and width from the spec, and the default network accepts another image size without a change.
- Camera mount and field of view: `CAMERA_X`, `CAMERA_Z`, `CAMERA_PITCH`, `CAMERA_FOV` at the top of `carla_env.py`.
- A new sensor type: add `add_<name>_camera` and `process_<name>_img` to `carla_env.py`, extend the `camera_type` branches in `reset()` and `step()`, and add the name to the `--camera` choices in `train_a3c.py`. Set `IMAGE_CHANNELS` in `rl_configuration.py` when the channel count changes. `add_depth_camera` is an unfinished starting point and is not connected to `--camera`.

## A new map

1. Copy `town03.py` to `<map>.py` with the lowercased map name (`Town04` becomes `town04.py`).
2. Replace the spawn and goal indices. Keep the shape of the `SCENARIOS` dict (`docs/environment.md`).
3. Run with `--map-name Town04 --scenario <id>`, or set `CARLA_MAP` in `.env`.

- A missing file or scenario id is an error.
- Scenario ids 1 and 2 run without middle goals on turns (`CarlaEnv._draw_optimal_route_lines` skips them), so use other ids for new scenarios.
- Commented tuples inside the scenario lists of `town03.py` are unused route variants. Keep them there.
- Add the new module to `py-modules` in `pyproject.toml` if you install the template as a package.
