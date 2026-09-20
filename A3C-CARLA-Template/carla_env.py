"""CARLA environment used by the A3C wrapper.

Connects to an already-running server, loads the selected map and scenario,
and exposes reset/step around cameras, collisions, lane invasions, and reward.
"""

import random
import time
import math
import importlib
import torch
from utils import ColoredPrint
from rl_configuration import Actions as ac
from rl_configuration import reward_function, REWARD_FROM_MP, REWARD_FROM_TP
import queue
import settings

import carla
from carla import ColorConverter as cc
import numpy as np
import cv2

from carla_navigation.global_route_planner import GlobalRoutePlanner
from carla_navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from state_observer import StateObserver
from carla_navigation.local_planner import RoadOption


MAX_ATTEMPTS = 10
WAIT_TIME = 0.5  # seconds

DECISIONS_DICT = {
    RoadOption.LEFT: 0,
    RoadOption.STRAIGHT: 1,
    RoadOption.RIGHT: 2
}

# ---------------------------------------------------------------------------
# Spawn/goal tables live in ``<map_name.lower()>.py`` (default: town03.py).
# ---------------------------------------------------------------------------

CAMERA_FOV = "75"


def load_map_scenarios(map_name):
    """Import ``SCENARIOS`` from ``<map_name.lower()>.py`` next to this file."""
    module_name = str(map_name).lower()
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            "No scenario file for map {!r}. Add {}.py next to carla_env.py "
            "(copy town03.py and replace spawn/goal indices). "
            "Import error: {}".format(map_name, module_name, error)
        ) from error
    scenarios = getattr(module, "SCENARIOS", None)
    if not scenarios:
        raise ValueError(
            "Module {!r} has no SCENARIOS dict.".format(module_name))
    return scenarios


class CarlaEnv:
    """Synchronous CARLA client for one worker."""

    def __init__(self, scenario, action_space='discrete', resX=250, resY=250, camera='semantic', port=None,
                 host=None, map_name=None, manual_control=False, spawn_point=False, terminal_point=False,
                 mp_density=25, verbose=False):
        if port is None:
            port = settings.PORT
        if host is None:
            host = settings.CARLA_HOST
        if map_name is None:
            map_name = settings.MAP_NAME

        self.host = host
        self.port = port
        self.map_name = map_name
        self.step_limit = settings.STEP_COUNTER
        self.spawning_type = settings.SPAWNING_TYPE
        self.draw = settings.DRAW
        self.mp_reward = REWARD_FROM_MP
        self.tp_reward = REWARD_FROM_TP

        self.client = carla.Client(self.host, port)
        self.client.set_timeout(120.0)

        self.log = ColoredPrint()
        self.verbose = bool(verbose)

        # Warn if the Python API and the running server disagree.
        client_ver = self.client.get_client_version()
        server_ver = self.client.get_server_version()

        if client_ver == server_ver:
            self.log.success(f"Client version: {client_ver}, Server version: {server_ver}")
        else:
            self.log.warn(f"Client version: {client_ver}, Server version: {server_ver}")

        self.world = self.client.load_world(self.map_name)
        self._apply_sync_settings()



        self.camera_type = camera
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.scenario_list = scenario
        try:
            self.scenario = self.scenario_list[0]  # Single scenario
        except IndexError:
            self.scenario = False
        
        self.goal_points_index = 0
        self.sp = spawn_point
        self.tp = terminal_point
        self.middle_goals = []
        self.middle_goals_density = mp_density
        self._map_scenarios = None
        self._active_spec = None
        self._route_goal = None
        self.create_scenario(self.sp, self.tp, self.middle_goals_density)

        # True while the planned route is inside a turn (used for middle goals).
        self.is_junction = False

        self.spectator = self.set_spectator()
        self.goal_location_trans, self.goal_location_loc, self.route = self.plan_the_route()
        self.action_space = self.create_action_space(action_space)
        self.actor_list = []

        # Camera mount: slightly forward, eye height, pitched down 10 degrees.
        self.transform = carla.Transform(
            carla.Location(x=0.3, z=2.5),
            carla.Rotation(pitch=-10, yaw=0)
        )

        self.manual_control = manual_control
        self.control = carla.VehicleControl()
        self.resX = resX
        self.resY = resY

        self.collision_history_list = []
        self.invasion_history_list = []
        self.stat_reward_mp = []
        self.step_counter = 0

        # Cameras
        self.show_cam = settings.SHOW_CAM
        self.front_camera = None
        self.preview_camera = None
        self.preview_camera_enabled = False
        self.done = False
        self.speed = 0
        self.prev_speed = 0

        self.state_observer = StateObserver()

        self.planner = None
        self.number_of_resets = 0
        self.car_decisions = []
        self.spawn_points_index = 0
        

        self.walker = None
        self.walker_controller = None
     
    def _apply_sync_settings(self):
        """Enable synchronous simulation with a fixed timestep."""
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.1
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.world.apply_settings(self.settings)


    def _require_scenario(self):
        if self._map_scenarios is None:
            self._map_scenarios = load_map_scenarios(self.map_name)
        if self.scenario not in self._map_scenarios:
            raise ValueError(
                "Map {!r} has no scenario id {}. Known ids: {}. "
                "Copy town03.py to {}.py and add the scenario.".format(
                    self.map_name, self.scenario,
                    sorted(self._map_scenarios),
                    str(self.map_name).lower()))
        return self._map_scenarios[self.scenario]

    def _copy_spawn(self, spawn, dy=0):
        loc = spawn.location
        return carla.Transform(
            carla.Location(loc.x, loc.y + dy, loc.z), spawn.rotation)

    def _pick_route(self, spec, spawn_points):
        """Resolve spawn transform + goal from a scenario spec."""
        select = spec.get("select", "random")
        spawn_dy = spec.get("spawn_dy", 0)

        if spec.get("goal") == "random_other_spawn":
            indices = spec["spawn_indices"]
            spawn_idx = indices[random.randrange(0, len(indices))]
            goal_idx = spawn_idx
            while goal_idx == spawn_idx:
                goal_idx = random.randrange(0, len(spawn_points))
            return (
                self._copy_spawn(spawn_points[spawn_idx], spawn_dy),
                ("spawn", goal_idx),
                spawn_idx,
            )

        if "spawn_range" in spec:
            lo, hi = spec["spawn_range"]
            spawn_idx = random.randint(lo, hi)
            goals = spec["goal_xyz_list"]
            goal_xyz = goals[random.randrange(0, len(goals))]
            return (
                self._copy_spawn(spawn_points[spawn_idx], spawn_dy),
                ("xyz", tuple(goal_xyz)),
                spawn_idx,
            )

        routes = spec["routes"]
        n = len(routes)
        if select == "cycle":
            idx = self.goal_points_index % n
            self.goal_points_index = idx + 1
        elif select == "single":
            idx = 0
        else:
            idx = random.randrange(0, n)
            self.goal_points_index = idx
        spawn_key, goal = routes[idx]
        spawn = self._copy_spawn(spawn_points[spawn_key], spawn_dy)
        if isinstance(goal, int):
            route_goal = ("spawn", goal)
        else:
            route_goal = ("xyz", tuple(goal))
        return spawn, route_goal, spawn_key

    def create_scenario(self, sp, tp, mp_d):
        """Set ``spawn_point`` from explicit indices or the active scenario table."""
        spawn_points = self.map.get_spawn_points()
        if sp not in (None, False) and tp not in (None, False):
            self.spawn_point = self._copy_spawn(spawn_points[sp])
            self._route_goal = ("spawn", tp)
            self._active_spec = None
        else:
            spec = self._require_scenario()
            self._active_spec = spec
            spawn, route_goal, spawn_key = self._pick_route(spec, spawn_points)
            self.spawn_point = spawn
            self._route_goal = route_goal
            self.spawn_points_index = spawn_key
        self.spawn_point_loc = self.spawn_point.location

    def set_spectator(self, d=6.4):
        """Place the spectator camera above the spawn point (copy Location)."""
        loc = self.spawn_point.location
        location = carla.Location(loc.x, loc.y, loc.z + 40)
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(location, carla.Rotation(yaw=0, pitch=-70, roll=0)))
        return self.spectator

    def plan_the_route(self):
        """Trace spawn-to-goal, drop duplicate waypoints, extract turn maneuvers."""
        dao = GlobalRoutePlannerDAO(self.map, 1.0)
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()

        if self._route_goal is None:
            raise ValueError(
                "No route selected for map {!r} scenario {}".format(
                    self.map_name, self.scenario))
        kind, payload = self._route_goal
        if kind == "xyz":
            x, y, z = payload
            self.goal_location_loc = carla.Location(x=x, y=y, z=z)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)
        elif kind == "spawn":
            self.goal_location_trans = self.map.get_spawn_points()[payload]
            self.goal_location_loc = self.goal_location_trans.location
        else:
            raise ValueError("Unknown route goal kind {!r}".format(kind))

        self.route = self.planner.trace_route(self.spawn_point_loc, self.goal_location_loc)

        # CARLA sometimes repeats consecutive waypoints; drop those and keep turns only.
        _ = []
        # decisions = [el2 for el1, el2 in self.route]
        # decisions = [decisions[index] for index in range(len(decisions)) if index==0 or decisions[index] != decisions[index-1]]
        decisions = [el2 for el1, el2 in self.route]
        decisions = [decisions[index] for index in range(len(decisions)) if (index==0 or decisions[index] != decisions[index-1]) and decisions[index] != RoadOption.LANEFOLLOW and decisions[index] != RoadOption.CHANGELANELEFT and decisions[index] != RoadOption.CHANGELANERIGHT]
        decisions = [DECISIONS_DICT[el] for el in decisions]

        for i in range(len(self.route) - 1):
            current_point = self.route[i][0].transform
            next_point = self.route[i + 1][0].transform

            if current_point == next_point:
                pass
            else:
                _.append(self.route[i])

        # Append the last route point
        _.append(self.route[-1])

        self.route = _

        self._draw_optimal_route_lines(self.route, draw=self.draw)
        self.car_decisions = decisions
        self.car_decisions.append(1)
        if self.verbose:
            print(f"Maneuvers to make in this episode: {self.car_decisions}")


        return self.goal_location_trans, self.goal_location_loc, self.route

    def spawn_car(self, spawning_type=1, episode=None):
        """Spawn the ego vehicle at the current spawn point."""

        tesla = self.blueprint_library.filter('model3')[0]
        tesla.set_attribute('role_name', 'ego')
        if spawning_type==0:
            self.spawn_point = random.choice(self.route)[0].transform
        elif spawning_type == 1:
            # self.spawn_point.location.x -= 9
            pass
        elif spawning_type==2:
            if bool(episode%2):
                self.spawn_point = self.route[0][0].transform
            else:
                self.spawn_point = self.route[-120][0].transform
        # self.spawn_point.location.z = 20.0  # keep spawn above the ground
        
        for attempt in range(MAX_ATTEMPTS):
            self.vehicle = self.world.try_spawn_actor(tesla, self.spawn_point)
            if self.vehicle is not None:
                if self.verbose:
                    print(f"Vehicle spawned on attempt {attempt + 1}")
                break
            time.sleep(WAIT_TIME)
        else:
            raise RuntimeError("Failed to spawn the vehicle after several attempts.")

        self.actor_list.append(self.vehicle)

        return self.vehicle

    def create_action_space(self, action_space):
        """Resolve discrete action ids from ``rl_configuration.Actions``."""
        if action_space == 'discrete':
            self.action_space = [
                getattr(ac, name) for name in ac.ACTIONS_NAMES.values()]
            return self.action_space
        else:
            self.action_space = action_space
            return self.action_space

    def add_rgb_camera(self):
        """Attach an RGB camera and queue its frames."""
        rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{self.resX}")
        rgb_cam_bp.set_attribute("image_size_y", f"{self.resY}")
        rgb_cam_bp.set_attribute("fov", CAMERA_FOV)

        rgb_cam = self.world.spawn_actor(rgb_cam_bp, self.transform, attach_to=self.vehicle)
        self.actor_list.append(rgb_cam)
        # remove_pictures()
        self.image_queue = queue.Queue()
        rgb_cam.listen(self.image_queue.put)
        # rgb_cam.listen(lambda data: self.process_rgb_img(data))
        # rgb_cam.listen(lambda data: data.save_to_disk('A_to_B/camera_rgb_outputs/%06d.png' % data.frame) )


    def process_rgb_img(self, image):
        """Convert a raw RGB frame to a CHW float tensor."""
        # os.makedirs('A_to_B/camera_rgb_outputs/', exist_ok=True)
        # image.save_to_disk('A_to_B/camera_rgb_outputs/%06d.png' % image.frame)


        i = np.array(image.raw_data)
        # Drop the unused alpha channel.
        i2 = i.reshape((self.resY, self.resX, 4))
        i3 = i2[:, :, :3]

        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)

        self.front_camera = torch.from_numpy(i3.copy()).permute(
            2, 0, 1).contiguous().unsqueeze(0).float()

    def add_semantic_camera(self):
        """Attach a semantic-segmentation camera and queue its frames."""
        semantic_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        # semantic_cam_bp = carla.sensor.Camera('MyCamera', PostProcessing='SemanticSegmentation')
        semantic_cam_bp.set_attribute('image_size_x', f'{self.resX}')
        semantic_cam_bp.set_attribute('image_size_y', f'{self.resY}')
        semantic_cam_bp.set_attribute('fov', CAMERA_FOV)

        # new_location = carla.Location(self.transform.location.x, 
        #                             self.transform.location.y, 
        #                             self.transform.location.z + 1.0)  # raise camera 1 m

        # new_rotation = carla.Rotation(self.transform.rotation.pitch - 15,  # pitch down 15 degrees
        #                             self.transform.rotation.yaw, 
        #                             self.transform.rotation.roll)

        # new_transform = carla.Transform(new_location, new_rotation)

        # semantic_cam_sensor = self.world.spawn_actor(semantic_cam_bp, new_transform, attach_to=self.vehicle)
        semantic_cam_sensor = self.world.spawn_actor(semantic_cam_bp, self.transform, attach_to=self.vehicle)

        # semantic_cam_sensor.listen(lambda data: self.process_semantic_img(data))
        self.actor_list.append(semantic_cam_sensor)

        self.image_queue = queue.Queue()
        semantic_cam_sensor.listen(self.image_queue.put)

    def process_semantic_img(self, image):
        """Convert a semantic frame to CityScapes colors and a CHW tensor."""
        image.convert(cc.CityScapesPalette)
        # image.convert(carla.ColorConverter.CityScapesPalette)
        # image = image.to_array() 
        image = np.array(image.raw_data)
        image = image.reshape((self.resY, self.resX, 4))
        image = image[:, :, :3]
        if self.show_cam:
            cv2.imshow("", image)
            cv2.waitKey(1)

        self.front_camera = torch.from_numpy(image.copy()).permute(
            2, 0, 1).contiguous().unsqueeze(0).float()

    def add_depth_camera(self):
        """Attach a depth camera."""
        depth_cam_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_cam_bp.set_attribute('image_size_x', f'{self.resX}')
        depth_cam_bp.set_attribute('image_size_y', f'{self.resY}')
        depth_cam_bp.set_attribute('fov', '75')

        depth_cam_sensor = self.world.spawn_actor(depth_cam_bp, self.transform, attach_to=self.vehicle)

        depth_cam_sensor.listen(lambda data: self.process_depth_img(data, "linear"))  # "linear" or "log"
        self.actor_list.append(depth_cam_sensor)

    def process_depth_img(self, image, lin_or_log="log"):
        """Convert a depth frame to metres. Output format is not standardized yet."""

        if lin_or_log == "linear":
            image.convert(cc.Depth)
        elif lin_or_log == "log":
            image.convert(cc.LogarithmicDepth)
        else:
            self.log.err(f"Wrong value of an lin_or_log argument, replace: {lin_or_log} with 'linear' or 'log'")
            return

        # Array of BGRA 32-bit pixels.
        image = np.array(image.raw_data)
        image = image.reshape((self.resX, self.resY, 4))
        image = image[:, :, :3]

        gray_depth_img = []

        for i in image:
            for j in i:
                b = j[0]
                g = j[1]
                r = j[2]
                normalized = (r + g * 200 + b * 200 * 200) / (200 * 200 * 200 - 1)
                in_meters = 1000 * normalized
                gray_depth_img.append(in_meters)

        if self.show_cam:
            cv2.imshow("", gray_depth_img)
            cv2.waitKey(1)

    def add_collision_sensor(self):
        """Attach a collision sensor on the ego vehicle."""
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        col_sensor_bp = self.world.spawn_actor(col_sensor_bp, self.transform, attach_to=self.vehicle)
        col_sensor_bp.listen(lambda data: self.collision_data_registering(data))
        self.actor_list.append(col_sensor_bp)

    def collision_data_registering(self, event):
        """Append a collision event to history."""
        coll_type = event.other_actor.type_id
        self.collision_history_list.append(event)

    def add_line_invasion_sensor(self):
        """Attach a lane-invasion sensor on the ego vehicle."""
        inv_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        inv_sensor_bp = self.world.spawn_actor(inv_sensor_bp, self.transform, attach_to=self.vehicle)
        inv_sensor_bp.listen(lambda data: self.invasion_data_registering(data))
        self.actor_list.append(inv_sensor_bp)

    def invasion_data_registering(self, invasion):
        """Append a lane-invasion event to history."""
        self.invasion_history_list.append(invasion)

    def _draw_optimal_route_lines(self, route, draw):
        """Collect middle-goal waypoints; optionally draw the planned route."""

        for i in range(0, len(route) - 1):
            # A yaw jump > 1 degree marks the start/end of a turn.
            curr_point_diff = abs(route[i + 1][0].transform.rotation.yaw - route[i][0].transform.rotation.yaw)
            diff = False

            if curr_point_diff > 1:
                diff = True

            # Start of a turn.
            if diff and i:
                if not self.is_junction:
                    if self.scenario not in [1, 2]:
                        self.is_junction = True
                        self.middle_goals.append(route[i][0])
            # End of a turn.
            elif self.is_junction and not diff:
                if self.scenario not in [1, 2]:
                    self.is_junction = False
                    self.middle_goals.append(route[i][0])

            if draw:
                self.world.debug.draw_line(route[i][0].transform.location, route[i + 1][0].transform.location,
                                         thickness=0.3, color=carla.Color(0, 0, 255), life_time=-1)

        # Waypoints become transforms from here on.
        for i, mg in enumerate(self.middle_goals):
            try:
                mg = mg.transform
                self.middle_goals[i] = mg
            except AttributeError:
                pass

        self.middle_goals.append(self.goal_location_trans)

        if draw:
            self.world.debug.draw_line(route[-1][0].transform.location, self.middle_goals[-1].location,
                                       thickness=0.3, color=carla.Color(0, 0, 255), life_time=-1)

        # Extra middle goals on long straights (segment longer than density).
        add_middle_goals = []

        for i, middle_goal in enumerate(self.middle_goals):

            length = len(self.middle_goals)

            if i == 0:
                # From spawn point to the first middle goal
                distance = self._calculate_distance_transform(self.spawn_point, middle_goal)
                factor = math.floor(distance / self.middle_goals_density)
                add_middle_goals.append([self.spawn_point, middle_goal, factor])

                # The first turn
                if length > 1:
                    distance = self._calculate_distance_transform(middle_goal, self.middle_goals[1])
                    factor = math.floor(distance / self.middle_goals_density)
                    add_middle_goals.append([middle_goal, self.middle_goals[1], factor])

            # Every other middle-goal to next-goal (or terminal) segment.
            elif i != length - 1:
                distance = self._calculate_distance_transform(middle_goal, self.middle_goals[i + 1])
                factor = math.floor(distance / self.middle_goals_density)
                add_middle_goals.append([middle_goal, self.middle_goals[i + 1], factor])

        self.middle_points(add_middle_goals)
        self.stat_reward_mp = []
        self._apply_middle_patches()

        # Draw middle points
        for middle_goal in self.middle_goals:
            # Counter 0 means this middle goal has not yet paid its reward.
            self.stat_reward_mp.append([middle_goal.location, 0])

    def _apply_middle_patches(self):
        """Apply optional spawn-table patches to ``middle_goals``."""
        spec = self._active_spec or {}
        for patch in spec.get("middle_patches") or []:
            op = patch[0]
            if op == "set":
                _, index, xyz = patch
                self.middle_goals[index] = carla.Transform(
                    carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))
            elif op == "insert":
                _, index, xyz = patch
                self.middle_goals.insert(
                    index, carla.Transform(
                        carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])))
            elif op == "append_copy":
                _, index = patch
                self.middle_goals.append(self.middle_goals[index])

    def spawn_npc_vehicle(self, spawn_index=50):
        """Spawn an autopilot vehicle at ``spawn_index``."""
        npc_bp = self.blueprint_library.filter("vehicle.*")[0]
        npc_bp.set_attribute("role_name", "autopilot")

        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) <= spawn_index:
            print(f"Not enough spawn points. Max: {len(spawn_points)}")
            return

        spawn_transform = spawn_points[spawn_index]
        spawn_transform.location.z += 0.5  # avoid colliding with the ground

        npc_vehicle = self.world.try_spawn_actor(npc_bp, spawn_transform)
        if npc_vehicle is None:
            print("Failed to spawn NPC vehicle.")
            return

        npc_vehicle.set_autopilot(True)
        self.actor_list.append(npc_vehicle)
        print(f"NPC vehicle spawned at spawn point {spawn_index}.")


    def spawn_single_pedestrian(self):
        """Spawn one pedestrian with an AI walker controller."""

        walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        walker_bp = random.choice(walker_blueprints)

        spawn_points = self.map.get_spawn_points()
        # if not spawn_points:
        #     print("No spawn points available.")
        #     return

        # spawn_transform = random.choice(spawn_points)
        spawn_transform = spawn_points[48]
        # spawn_transform.location.z += 1.0  # lift slightly to avoid ground collision

        self.walker = self.world.try_spawn_actor(walker_bp, spawn_transform)
        if self.walker is None:
            print("Failed to spawn pedestrian.")
        else:
            self.walker.set_simulate_physics(True)
            self.actor_list.append(self.walker)
            self.walker_controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=self.walker)
        if self.walker_controller is not None:
            self.actor_list.append(self.walker_controller)

        self.walker_controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=self.walker)
        self.walker_controller.start()
        self.world.tick()

        if self.walker_controller.is_alive:
            print("Walker spawned but has no destination yet.")
            # destination = self.world.get_random_location_from_navigation()
            # if destination is not None:
            #     self.walker_controller.go_to_location(destination)
            #     self.walker_controller.set_max_speed(random.uniform(0.5, 1.5))
            # else:
            #     print("Nie znaleziono celu dla pieszego.")

    @staticmethod
    def _calculate_distance_transform(current_location, goal_location):
        """Euclidean distance between two CARLA transforms."""

        distance = math.sqrt((goal_location.location.x - current_location.location.x) ** 2 +
                             (goal_location.location.y - current_location.location.y) ** 2 +
                             (goal_location.location.z - current_location.location.z) ** 2)

        return distance

    @staticmethod
    def _calculate_distance_locations(current_location, goal_location):
        """Euclidean distance between two CARLA locations."""

        distance = math.sqrt((goal_location.x - current_location.x) ** 2 +
                             (goal_location.y - current_location.y) ** 2 +
                             (goal_location.z - current_location.z) ** 2)

        return distance

    def middle_points(self, middle_points_list):
        """Insert extra middle goals on segments longer than ``middle_goals_density``."""

        for middle_point in middle_points_list:
            mp_list = [middle_point[0], middle_point[1]]
            factor = middle_point[2]
            if factor > 0:
                self.calculate_middle_points(mp_list, factor)

    def calculate_middle_points(self, mp_list, factor):
        """Bisect a segment ``factor`` times and insert the new points into ``middle_goals``."""
        middle_points_len = len(mp_list)

        while factor:

            new_mp = []

            for i in range(middle_points_len - 1):
                location = carla.Location((mp_list[i].location.x + mp_list[i + 1].location.x) / 2,
                                        (mp_list[i].location.y + mp_list[i + 1].location.y) / 2,
                                        (mp_list[i].location.z + mp_list[i + 1].location.z) / 2)

                angle = carla.Rotation(mp_list[i + 1].rotation.pitch, mp_list[i + 1].rotation.yaw,
                                       mp_list[i + 1].rotation.roll)

                middle = carla.Transform(location, angle)
                new_mp.append(middle)

            for i, v in enumerate(new_mp):
                mp_list.insert(2 * i + 1, v)

            factor -= 1

            if factor:
                return self.calculate_middle_points(mp_list, factor)
            else:
                # Snap z to the nearest route waypoint when xy is close (planner z can sit under the map).
                for mp in new_mp:
                    for r in self.route:
                        r = r[0].transform.location

                        x_diff = abs(mp.location.x - r.x)
                        y_diff = abs(mp.location.y - r.y)
                        diff_sum = x_diff + y_diff

                        if diff_sum < 1:
                            mp.location.z = r.z

                for mp in new_mp:
                    index = self.middle_goals.index(mp_list[-1])
                    self.middle_goals.insert(index, mp)

    def calculate_distance(self):
        """Return ``(distance_to_goal, vehicle_location)``."""
        vehicle_location = self.vehicle.get_location()
        distance = math.sqrt((self.goal_location_loc.x - vehicle_location.x) ** 2 +
                             (self.goal_location_loc.y - vehicle_location.y) ** 2 +
                             (self.goal_location_loc.z - vehicle_location.z) ** 2)

        if self.draw:
            self.world.debug.draw_string(
                vehicle_location, "X", life_time=100, persistent_lines=True)
        return distance, vehicle_location

    def calculate_speed(self):
        """Ego speed in km/h."""
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        return speed

    def car_control_continuous(self, action):
        """Apply throttle/steer/brake from a two-value continuous action."""
        gas_value = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        self.control.throttle = gas_value
        self.control.steer = steer
        self.control.brake = brake
        self.control.hand_brake = False
        self.control.reverse = False
        self.control.manual_gear_shift = False
        self.vehicle.apply_control(self.control)

    def car_control_discrete(self, action):
        """Apply throttle/steer/brake from a discrete action id."""
        self.control.throttle = ac.ACTION_CONTROL[self.action_space[action]][0]
        self.control.brake = ac.ACTION_CONTROL[self.action_space[action]][1]
        self.control.steer = ac.ACTION_CONTROL[self.action_space[action]][2]
        self.control.hand_brake = False
        self.control.reverse = False
        self.control.manual_gear_shift = False
        self.vehicle.apply_control(self.control)

    def calculate_route_distance(self, current_location):
        """Shortest distance from the vehicle to any planned-route waypoint."""
        route_distance = min([self._calculate_distance_locations(current_location, x[0].transform.location)
                              for x in self.route])

        return route_distance

    def static_reward_mp(self, vehicle_location, static_reward_from_mp):
        """Pay a one-shot reward when the vehicle first reaches a middle goal."""
        done = False

        mp_distances = [self._calculate_distance_locations(vehicle_location, x[0]) for x in self.stat_reward_mp]
        mp_min = min(mp_distances)
        mp_index = mp_distances.index(mp_min)

        if mp_min < 3 and self.stat_reward_mp[mp_index][1] == 0:
            self.stat_reward_mp[mp_index][1] = 1
            if mp_index == len(self.stat_reward_mp) - 1:
                return static_reward_from_mp, True
            else:
                return static_reward_from_mp, done
        else:
            return 0, done
    


    def reload_world(self):
        """Reload the CARLA world and reset episode-local state."""
        self.destroy_agents()
        self.actor_list = []


        old_world = self.client.get_world()
        if old_world is not None:
            prev_world_id = old_world.id
            del old_world
        else:
            prev_world_id = None
        

        self.world = self.client.reload_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        
        spawn_points = self.world.get_map().get_spawn_points()
        if self.draw:
            for i, spawn_point in enumerate(spawn_points):
                location = spawn_point.location
                self.world.debug.draw_string(
                    location, str(i), draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0), life_time=120.0)
        
        self._apply_sync_settings()

        # self.world = self.client.reload_world()



        # Duplicate of _apply_sync_settings (kept for reference).
        # self.settings = self.world.get_settings()
        # self.settings.synchronous_mode = True
        # self.settings.fixed_delta_seconds = 0.1
        # self.settings.max_substep_delta_time = 0.01
        # self.settings.max_substeps = 10
        # self.world.apply_settings(self.settings)
        # self.client.reload_world(False)

        tries = 3
        self.world = self.client.get_world()

        # spawn_points = self.world.get_map().get_spawn_points()
        # for i, spawn_point in enumerate(spawn_points):
        #     location = spawn_point.location
        #     self.world.debug.draw_string(location, str(i), draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=120.0)
            # self.world.tick()

        while prev_world_id == self.world.id and tries > 0:
            tries -= 1
            self.world.tick()
            self.world = self.client.get_world()
        # self.world = self.client.reload_world()

        self.collision_history_list = []
        self.invasion_history_list = []
        self.middle_goals = []
        self.step_counter = 0
        self.stat_reward_mp = []
        self.front_camera = None
        self.preview_camera = None
        self.preview_camera_enabled = False
        self.is_junction = False
        self.done = False
        self.speed = 0

    def reset_episode_state(self):
        """Clear actors and episode counters without reloading the world."""
        self.destroy_agents()
        self.actor_list = []
        if hasattr(self, 'image_queue'):
            while not self.image_queue.empty():
                _ = self.image_queue.get()
        self.collision_history_list = []
        self.invasion_history_list = []
        self.middle_goals = []
        self.step_counter = 0
        self.stat_reward_mp = []
        self.front_camera = None
        self.preview_camera = None
        self.preview_camera_enabled = False
        self.is_junction = False
        self.done = False
        self.speed = 0
        self.prev_speed = 0

    def _get_latest_camera_image(self, timeout=2.0):
        """Drain the camera queue; raise if no frame arrives in time."""
        try:
            image = self.image_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("time-out waiting for camera image")
        while not self.image_queue.empty():
            image = self.image_queue.get()
        return image

    def reset(self, episode, save_image=False, reload_world=True):
        """Rebuild the episode and return ``(observation, speed)``."""
        if reload_world:
            self.reload_world()
        else:
            self.reset_episode_state()
    
        
        if self.scenario:
            self.scenario = random.choice(self.scenario_list)
            self.create_scenario(self.sp, self.tp, self.middle_goals_density)

        # self.spawn_single_pedestrian()
        # self.spawn_npc_vehicle(spawn_index=48)

        self.plan_the_route()
        self.spawn_car(self.spawning_type, episode)
        self.set_spectator()

        if self.camera_type == 'rgb':
            self.add_rgb_camera()
        elif self.camera_type == 'semantic':
            self.add_semantic_camera()
        else:
            self.log.err(f"Wrong camera type. Pick rgb or semantic, not: {self.camera_type}")

        # self.add_depth_camera()
        self.add_collision_sensor()
        self.add_line_invasion_sensor()

        # self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        # time.sleep(2)
        # time.sleep(0.5)

        # while self.front_camera is None:
        #     time.sleep(0.01)

        # self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        # results_queue.put(1)
        # # A frame from the spawn point
        # while not self.image_queue.empty():
        #     _ = self.image_queue.get()

        # A few ticks so the debug route is visible on the first frame.
        self.step_apply_action(3)
        for i in range(15):
            self.world.tick()
            
        while not self.image_queue.empty():
            _ = self.image_queue.get()

        self.world.tick()
        image = self._get_latest_camera_image(timeout=2.0)

        self.state_observer.image = image

        # if save_image:
        #     self.state_observer.save_to_disk(image, episode, 0)
        
        if self.camera_type == 'rgb':
            self.process_rgb_img(image)
        else:
            self.process_semantic_img(image)
        return self.front_camera, float(self.speed)
    
    def step_apply_action(self, action):
        """Apply one control command without ticking the world."""
        self.step_counter += 1

        if self.action_space == 'continuous':
            self.car_control_continuous(action)
        else:
            self.car_control_discrete(action)

    def step(self, episode, step, save_image=False, on_junction=False):
        """Observe reward and the next camera frame after the world has been ticked."""
        # self.step_counter += 1

        # if self.action_space == 'continuous':
        #     self.car_control_continuous(action)
        # else:
        #     self.car_control_discrete(action)

        
        # if sleep_time:
        #     time.sleep(sleep_time)

        distance_from_goal, vehicle_location = self.calculate_distance()

        route_distance = self.calculate_route_distance(vehicle_location)
        self.speed = self.calculate_speed()
        speed_value = float(self.speed)

        static_reward_from_mp = self.mp_reward
        mp_static_reward, self.done = self.static_reward_mp(vehicle_location, static_reward_from_mp)
                
        if self.done:
            terminal_state_reward = self.tp_reward
        else:
            terminal_state_reward = 0

        # One invasion event can flood the history; count at most one per step.
        if len(self.invasion_history_list) != 0:
            invasion_counter = 1
        else:
            invasion_counter = 0
        self.last_invasion_counter = invasion_counter
        self.invasion_history_list = []

        reward, done = reward_function(self.collision_history_list, invasion_counter, self.speed, route_distance,
                                             mp_static_reward, terminal_state_reward, on_junction, self.prev_speed)

        self.prev_speed = self.speed
        self.done = self.done or done

        if self.step_counter >= self.step_limit:
            self.done = True
        image = self._get_latest_camera_image(timeout=2.0)
        self.state_observer.image = image

        # if save_image:
        #     self.state_observer.save_to_disk(image, episode, step)

        if self.camera_type == 'rgb':
            self.process_rgb_img(image)
        else:
            self.process_semantic_img(image)
        return self.front_camera, reward, self.done, route_distance, speed_value, distance_from_goal

    def destroy_agents(self):
        """Stop sensors and destroy spawned actors."""
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()
        
