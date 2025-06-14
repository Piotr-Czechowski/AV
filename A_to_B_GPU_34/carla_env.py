import glob
import os
import pdb
import random
#import wandb
import sys
import time
import math
import torch
import settings
import torch.multiprocessing as mp
from utils import ColoredPrint
from ACTIONS import ACTIONS as ac
import queue

# try:
#     sys.path.append(glob.glob(settings.CARLA_EGG_PATH % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
from carla import ColorConverter as cc
import numpy as np
import subprocess
import cv2

from utils import reward_function
from A_to_B_GPU_34.carla_navigation.global_route_planner import GlobalRoutePlanner
from A_to_B_GPU_34.carla_navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from settings import SHOW_CAM
from state_observer import StateObserver
from carla_navigation.local_planner import RoadOption


# Global settings
how_many_steps = settings.STEP_COUNTER
sleep_time = settings.SLEEP_BETWEEN_ACTIONS
mp_reward = settings.REWARD_FROM_MP
tp_reward = settings.REWARD_FROM_TP
serv_resx = settings.SERV_RESX
serv_resy = settings.SERV_RESY
port = settings.PORT
spawning_type = settings.SPAWNING_TYPE
logging = settings.LOGGING
draw = settings.DRAW

MAX_ATTEMPTS = 10
WAIT_TIME = 0.5  # sekundy

DECISIONS_DICT = {
    RoadOption.LEFT: 0,
    RoadOption.STRAIGHT: 1,
    RoadOption.RIGHT: 2
}

MAP_POINTS_SC10 = [(50, 203.913498, 0.275307),
              (100, 203.788742, 1.3),
              (-55.387177, 0.558450, 0.0),
              (-105.387177, -3.140184, 0.0),
              (74, -40, 1.0),
              (-6.5, -44, 0.0)]

MAP_POINTS_SC11 = [ (14,(-55.387177, 0.558450, 0.0)),
                    (14,(-105.387177, -3.140184, 0.0)),]

def start_carla_server(args):
    return subprocess.Popen(f'CarlaUE4.exe ' + args, cwd=settings.CARLA_PATH, shell=True)

def remove_pictures():
    folder_path = 'A_to_B_GPU_34/camera_rgb_outputs'
    if not os.path.isdir(folder_path):
        return
    # Iteruj przez wszystkie pliki w folderze i usuń je
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Usunięcie pliku
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Usunięcie pustego folderu (opcjonalnie)
        except Exception as e:
            print(f'Nie można usunąć {file_path}. Powód: {e}')


class CarlaEnv:
    """
    Create Carla environment
    """
    def __init__(self, scenario, action_space='discrete', resX=200, resY=200, camera='semantic', port=port,
                 manual_control=False, spawn_point=False, terminal_point=False, mp_density=25):
        # Run the server on 127.0.0.1/port
        #start_carla_server(f'-windowed -carla-server -fps=60 -ResX={serv_resx} -ResY={serv_resy} -quality-level=Low '
        #                   f'-carla-world-port={port}')
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(30.0)
        
        #for debugging
        # Enable to use colors
        self.log = ColoredPrint()

        # Make sure that server and client versions are the same
        client_ver = self.client.get_client_version()
        server_ver = self.client.get_server_version()

        if client_ver == server_ver:
            self.log.success(f"Client version: {client_ver}, Server version: {server_ver}")
        else:
            self.log.warn(f"Client version: {client_ver}, Server version: {server_ver}")

        self.world = self.client.load_world('Town03')



        self.camera_type = camera
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.scenario_list = scenario
        try:
            self.scenario = self.scenario_list[0]  # Single scenario
        except IndexError:
            self.scenario = False

        self.sp = spawn_point
        self.tp = terminal_point
        self.middle_goals = []
        self.middle_goals_density = mp_density  # how dense middle points should be?
        self.create_scenario(self.sp, self.tp, self.middle_goals_density)

        # Do we have a junction? Used in recognizing junctions for setting middle goals
        self.is_junction = False

        # Set the spectator
        self.spectator = self.set_spectator()

        # Plan the route to from the initial point to the terminal point, draw that route
        self.goal_location_trans, self.goal_location_loc, self.route = self.plan_the_route()

        # Environment possible actions
        self.action_space = self.create_action_space(action_space)

        # List of all actors in the environment
        self.actor_list = []

        self.transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        # self.transform = carla.Transform(
        #     carla.Location(x=1.2, z=2.5),  # Kamera lekko z przodu, na wysokości ~oczu
        #     carla.Rotation(pitch=-10)      # Pochylona w dół o 10 stopni
        # )

        self.manual_control = manual_control
        # if not manual_control:
        #     self.vehicle = self.spawn_car(spawning_type)

        # Manages the basic movement of a vehicle using typical driving controls
        self.control = carla.VehicleControl()

        # Images X, Y resolutions
        self.resX = resX
        self.resY = resY

        # Variables which have to reset at the end of each episode
        self.collision_history_list = []
        # History of crossing a lane markings
        self.invasion_history_list = []
        self.stat_reward_mp = []

        # The number of steps in one episode
        self.step_counter = 0

        # Cameras
        self.show_cam = SHOW_CAM
        self.front_camera = None
        self.preview_camera = None
        self.preview_camera_enabled = False
        self.done = False
        self.speed = 0

        self.state_observer = StateObserver()

        self.planner = None
        self.number_of_resets = 0
        self.car_decisions = []
        self.spawn_points_index = 0
        self.goal_points_index = 0

        self.walker = None
        self.walker_controller = None
     

    def create_scenario(self, sp, tp, mp_d):

        if sp and tp:
            # Usage of spawn_point and terminal_point
            self.spawn_point = self.map.get_spawn_points()[sp]

        elif self.scenario in [1, 2]:
            # Straight short line - 1
            # Straight long line - 2
            self.spawn_point = self.map.get_spawn_points()[3]

        elif self.scenario == 3:
            # Right turn
            sp = self.map.get_spawn_points()[11]
            sp.location.y -= 11
            self.spawn_point = carla.Transform(sp.location, sp.rotation)

        elif self.scenario == 4:
            # Left turn
            sp = self.map.get_spawn_points()[12]
            sp.location.y -= 15
            self.spawn_point = carla.Transform(sp.location, sp.rotation)

        elif self.scenario == 5:
            # Little straight line and right turn
            sp = self.map.get_spawn_points()[11]
            sp.location.y += 10
            self.spawn_point = carla.Transform(sp.location, sp.rotation)

        elif self.scenario == 6:
            # Little straight line and left turn
            sp = self.map.get_spawn_points()[12]
            # sp.location.y -= 20
            self.spawn_point = carla.Transform(sp.location, sp.rotation)

        elif self.scenario in (7, 8):
            # Long straight line and 2 right turns
            # self.spawn_point = self.map.get_spawn_points()[57]
            self.spawn_point = self.map.get_spawn_points()[130]
            # self.spawn_point = carla.Transform(sp.location, sp.rotation)
            self.goal_point = 122

        elif self.scenario == 10:
            self.spawn_points_index = random.randint(0,30)
            sp = self.map.get_spawn_points()[self.spawn_points_index]
            self.spawn_point = carla.Transform(sp.location, sp.rotation)

        elif self.scenario == 11:
            try:
                self.goal_points_index = random.randint(0,len(MAP_POINTS_SC11)-1)
                sp_number = MAP_POINTS_SC11[self.goal_points_index][0]
                sp = self.map.get_spawn_points()[sp_number]
                self.spawn_point = carla.Transform(sp.location, sp.rotation)
            except TypeError:
                print(3)
        else:
            self.log.err(f"Invalid params: scenario: {self.scenario} or sp: {sp}, tp:{tp},"
                         f" mp_d:{mp_d}")

        self.spawn_point_loc = self.spawn_point.location

    def set_spectator(self, d=6.4):
        """
        Get specator's camera angles
        :param d: constant
        :return: self.spectator - spectator's exact location and its angles
        """
        # angle = 90  # cos and sin argument

        # spectator_coordinates = carla.Location(self.spawn_point_loc.x,
        #                                        self.spawn_point_loc.y,
        #                                        self.spawn_point_loc.z)

        # if self.scenario == 7:
        #     spectator_coordinates.x -= 3
        #     spectator_coordinates.y += 10
        #     spectator_coordinates.z += 50
        #     pass
        # else:
        #     spectator_coordinates.x += 10
        #     spectator_coordinates.y += 10
        #     spectator_coordinates.z += 50

        # a = math.radians(angle)
        # location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + spectator_coordinates

        # self.spectator = self.world.get_spectator()
        # """
        # yaw - rotating your vision in 2D (left <-, right ->)  
        # pitch - looking more to the sky or the road 
        # roll - leaning your vision (e.g. from | to ->)
        # """
        # self.spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=-60, pitch=-60, roll=0)))

        # return self.spectator
        spectator_coordinates = self.spawn_point.location
        location = spectator_coordinates
        location.z = location.z + 30
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=0, pitch=-90, roll=0)))
        return self.spectator

    def plan_the_route(self):
        """
        plan the route between the initial point and the terminal point then draw it
        :param: changes -
        :return: goal_location - terminal point, self.route - list of tuples of
        """
        # Plan a route to the destination
        way_points = self.map.generate_waypoints(2.0)
        dao = GlobalRoutePlannerDAO(self.map, 2.0)
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()

        if self.scenario == 1:
            self.goal_location_loc = carla.Location(x=50, y=203.913498, z=0.275307)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)

        elif self.scenario == 2:
            self.goal_location_loc = carla.Location(x=100, y=203.788742, z=1.3)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)

        elif self.scenario in [3, 5]:
            self.goal_location_loc = carla.Location(x=-55.387177, y=0.558450, z=0.0)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)

        elif self.scenario in [4, 6]:
            self.goal_location_loc = carla.Location(x=-105.387177, y=-3.140184, z=0.0)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)
        elif self.scenario == 8:
            self.goal_location_loc = carla.Location(x=74, y=-40, z=1.0)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)
        elif self.scenario == 10:
            self.goal_points_index = random.randint(0,len(MAP_POINTS_SC10)-1)
            x, y, z = MAP_POINTS_SC10[self.goal_points_index]
            self.goal_location_loc = carla.Location(x=x, y=y, z=z)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)
        elif self.scenario == 11:
            x, y, z = MAP_POINTS_SC11[self.goal_points_index][1]
            self.goal_location_loc = carla.Location(x=x, y=y, z=z)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)

        
        else:
            # self.goal_location_loc = way_points[self.goal_point].transform.location
            # self.goal_location_trans = way_points[self.goal_point].transform
            self.goal_location_loc = carla.Location(x=-6.5, y=-44, z=0.0)
            self.goal_location_trans = carla.Transform(self.goal_location_loc)

        self.route = self.planner.trace_route(self.spawn_point_loc, self.goal_location_loc)

        # Delete duplicates (for some reason there are duplicates in self.route)
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

        self._draw_optimal_route_lines(self.route, draw=draw)
        self.car_decisions = decisions
        self.car_decisions.append(1)
        print(f"Manouvers to make in this episode: {self.car_decisions}")


        return self.goal_location_trans, self.goal_location_loc, self.route

    def spawn_car(self, spawning_type=1, episode=None):
        """
        Spawn a car
        :return: vehicle
        """

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
        # self.spawn_point.location.z = 20.0 # żeby nie był za nisko
        
        for attempt in range(MAX_ATTEMPTS):
            self.vehicle = self.world.try_spawn_actor(tesla, self.spawn_point)
            if self.vehicle is not None:
                print(f"Pojazd zespawnowany w próbie {attempt + 1}")
                break
            time.sleep(WAIT_TIME)
        else:
            raise RuntimeError("Nie udało się zespawnować pojazdu po kilku próbach.")

        self.actor_list.append(self.vehicle)

        return self.vehicle

    def create_action_space(self, action_space):
        if action_space == 'discrete':
            self.action_space = [getattr(ac, action) for action in settings.ACTIONS]
            return self.action_space
        else:
            self.action_space = action_space
            return self.action_space

    def add_rgb_camera(self):
        """
        Attach RGB camera to the vehicle
        The "RGB" camera acts as a regular camera capturing images from the scene
        """
        rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{self.resX}")
        rgb_cam_bp.set_attribute("image_size_y", f"{self.resY}")
        rgb_cam_bp.set_attribute("fov", f"110")

        rgb_cam = self.world.spawn_actor(rgb_cam_bp, self.transform, attach_to=self.vehicle)
        self.actor_list.append(rgb_cam)
        # remove_pictures()
        self.image_queue = queue.Queue()
        rgb_cam.listen(self.image_queue.put)
        # rgb_cam.listen(lambda data: self.process_rgb_img(data))
        # rgb_cam.listen(lambda data: data.save_to_disk('A_to_B/camera_rgb_outputs/%06d.png' % data.frame) )


    def process_rgb_img(self, image):
        """
        Process RGB images
        :param image: raw data from the rgb camera
        :return:
        """
        # os.makedirs('A_to_B/camera_rgb_outputs/', exist_ok=True)
        # image.save_to_disk('A_to_B/camera_rgb_outputs/%06d.png' % image.frame)


        i = np.array(image.raw_data)
        # Also returns alfa values - not only rgb
        i2 = i.reshape((self.resY, self.resX, 4))
        i3 = i2[:, :, :3]

        if self.show_cam:
            # noinspection PyUnresolvedReferences
            cv2.imshow("", i3)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

        self.front_camera = torch.Tensor(i3).view(3, self.resX, self.resY).unsqueeze(0)

    def add_semantic_camera(self):
        """
        Attach semantic camera to the vehicle
        The "Semantic Segmentation" camera classifies every object in the view by displaying it in a different color
        according to the object class. E.g. pedestrians appear in a different color than vehicles.
        Original images are totally black
        """
        semantic_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        # semantic_cam_bp = carla.sensor.Camera('MyCamera', PostProcessing='SemanticSegmentation')
        semantic_cam_bp.set_attribute('image_size_x', f'{self.resX}')
        semantic_cam_bp.set_attribute('image_size_y', f'{self.resY}')
        semantic_cam_bp.set_attribute('fov', '110')

        # new_location = carla.Location(self.transform.location.x, 
        #                             self.transform.location.y, 
        #                             self.transform.location.z + 1.0)  # Podniesienie kamery o 1m

        # new_rotation = carla.Rotation(self.transform.rotation.pitch - 15,  # Nachylenie w dół o 15 stopni
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
        """
        Process semantic images
        :param image: raw data from the semantic camera
        """
        image.convert(cc.CityScapesPalette)
        # image.convert(carla.ColorConverter.CityScapesPalette)
        # image = image.to_array() 
        image = np.array(image.raw_data)
        image = image.reshape((self.resX, self.resY, 4))
        image = image[:, :, :3]
        if self.show_cam:
            # noinspection PyUnresolvedReferences
            cv2.imshow("", image)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

        # self.front_camera = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        self.front_camera = torch.Tensor(image).view(3, self.resX, self.resY).unsqueeze(0).float()

    def add_depth_camera(self):
        """
        The camera provides a raw data of the scene codifying the distance of each pixel to the camera
        (also known as depth buffer or z-buffer) to create a depth map of the elements.
        The image codifies depth value per pixel using 3 channels of the RGB color space,
        from less to more significant bytes: R -> G -> B.
        """
        depth_cam_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_cam_bp.set_attribute('image_size_x', f'{self.resX}')
        depth_cam_bp.set_attribute('image_size_y', f'{self.resY}')
        depth_cam_bp.set_attribute('fov', '90')

        depth_cam_sensor = self.world.spawn_actor(depth_cam_bp, self.transform, attach_to=self.vehicle)

        depth_cam_sensor.listen(lambda data: self.process_depth_img(data, "linear"))  # "linear" or "log"
        self.actor_list.append(depth_cam_sensor)

    # TODO standardization the data from a depth camera
    def process_depth_img(self, image, lin_or_log="log"):
        """

        :param image: image from depth camera
        :param lin_or_log: linear or logarithmic conversion
        """

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
            # noinspection PyUnresolvedReferences
            cv2.imshow("", gray_depth_img)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

    def add_collision_sensor(self):
        """
        This sensor, when attached to an actor, it registers an event each time the actor collisions against sth
        in the world. This sensor does not have any configurable attribute.
        """
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        col_sensor_bp = self.world.spawn_actor(col_sensor_bp, self.transform, attach_to=self.vehicle)
        col_sensor_bp.listen(lambda data: self.collision_data_registering(data))
        self.actor_list.append(col_sensor_bp)

    def collision_data_registering(self, event):
        """
        Register collisions
        :param event: data from the collision sensor
        """
        coll_type = event.other_actor.type_id
        self.collision_history_list.append(event)

    def add_line_invasion_sensor(self):
        """
        This sensor, when attached to an actor, it registers an event each time the actor crosses a lane marking.
        This sensor is somehow special as it works fully on the client-side.
        The lane invasion uses the road data of the active map to determine whether a vehicle is invading another lane.
        This information is based on the OpenDrive file provided by the map,
        therefore it is subject to the fidelity of the OpenDrive description. In some places there might be
        discrepancies between the lanes visible by the cameras and the lanes registered by this sensor.

        This sensor does not have any configurable attribute.
        """
        inv_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        inv_sensor_bp = self.world.spawn_actor(inv_sensor_bp, self.transform, attach_to=self.vehicle)
        inv_sensor_bp.listen(lambda data: self.invasion_data_registering(data))
        self.actor_list.append(inv_sensor_bp)

    def invasion_data_registering(self, invasion):
        """
        History of events when the actor crossed a lane marking
        :param invasion: - event
        """
        self.invasion_history_list.append(invasion)

    def _draw_optimal_route_lines(self, route, draw):
        """
        Draw optimal route between initial point and terminal point
        :param route: list of tuples
        """

        for i in range(0, len(route) - 1):
            """
            Get middle points from making a turn
            Calculate difference between angles of neighbour route points to detect turns
            Turns are when the yaw parameter changes (difference grater than 1) 
            """
            curr_point_diff = abs(route[i + 1][0].transform.rotation.yaw - route[i][0].transform.rotation.yaw)
            diff = False

            if curr_point_diff > 1:
                diff = True

            # The beginning of the turn
            if diff and i:
                if not self.is_junction:
                    if self.scenario not in [1, 2]:
                        self.is_junction = True
                        self.middle_goals.append(route[i][0])
            # The ending of the turn
            elif self.is_junction and not diff:
                if self.scenario not in [1, 2]:
                    self.is_junction = False
                    self.middle_goals.append(route[i][0])

            if draw:
                self.world.debug.draw_line(route[i][0].transform.location, route[i + 1][0].transform.location,
                                         thickness=0.3, color=carla.Color(0, 0, 255), life_time=-1)

        # We do not need self.middle_goals to be Waypoint class anymore we can change that to transform class
        for i, mg in enumerate(self.middle_goals):
            try:
                mg = mg.transform
                self.middle_goals[i] = mg
            except AttributeError:
                pass

        # Add terminal point
        self.middle_goals.append(self.goal_location_trans)

        # Do not need middle points anymore
        if draw:
            self.world.debug.draw_line(route[-1][0].transform.location, self.middle_goals[-1].location,
                                       thickness=0.3, color=carla.Color(0, 0, 255), life_time=-1)

        # Add middle points to long lines (when the size of long line > middle_goals_density)
        add_middle_goals = []

        # Let's calculate distances between points
        # We will use factor floor(distance / self.middle_goals_density) - to determine how many points should we add
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

            # Every other middle point -> middle point or middle point -> terminal point distnace
            elif i != length - 1:
                distance = self._calculate_distance_transform(middle_goal, self.middle_goals[i + 1])
                factor = math.floor(distance / self.middle_goals_density)
                add_middle_goals.append([middle_goal, self.middle_goals[i + 1], factor])

        # Get middle points for long lines
        self.middle_points(add_middle_goals)

        # Static reward from middle points
        self.stat_reward_mp = []

        if self.scenario == 3:
            self.middle_goals[0] = carla.Transform(carla.Location(x=-70.599335, y=1.434147, z=0.0))

        if self.scenario == 4:
            self.middle_goals[0] = carla.Transform(carla.Location(x=-91.646820, y=-2.737971, z=0.0))
            self.middle_goals[1] = carla.Transform(carla.Location(x=-83.688499, y=0.805027, z=0.0))
            self.middle_goals.append(self.middle_goals[2])
            self.middle_goals[2] = carla.Transform(carla.Location(x=-99.680237, y=-3.129901, z=0.0))

        if self.scenario == 5:
            self.middle_goals.insert(1, carla.Transform(carla.Location(x=-70.599335, y=1.434147, z=0.0)))

        if self.scenario == 6:
            self.middle_goals[1] = carla.Transform(carla.Location(x=-83.688499, y=0.805027, z=0.0))

        # Draw middle points
        for middle_goal in self.middle_goals:
            # if draw:
            #     self.world.debug.draw_point(middle_goal.location, size=0.15, life_time=-1)
            # Static reward for mp
            # Add each middle point with counter 0 which indicates if middle point has already given a reward
            self.stat_reward_mp.append([middle_goal.location, 0])

    def spawn_npc_vehicle(self, spawn_index=50):
        """
        Spawn AI-controlled NPC vehicle at specified spawn point.
        """
        npc_bp = self.blueprint_library.filter("vehicle.*")[0]  # pierwszy pojazd z listy
        npc_bp.set_attribute("role_name", "autopilot")

        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) <= spawn_index:
            print(f"Nie ma tylu punktów spawn. Max: {len(spawn_points)}")
            return

        spawn_transform = spawn_points[spawn_index]
        spawn_transform.location.z += 0.5  # uniknij kolizji z podłożem

        npc_vehicle = self.world.try_spawn_actor(npc_bp, spawn_transform)
        if npc_vehicle is None:
            print("Nie udało się zespawnować NPC.")
            return

        npc_vehicle.set_autopilot(True)
        self.actor_list.append(npc_vehicle)
        print(f"NPC pojazd zespawnowany w punkcie {spawn_index}.")


    def spawn_single_pedestrian(self):

        walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        walker_bp = random.choice(walker_blueprints)

        spawn_points = self.map.get_spawn_points()
        # if not spawn_points:
        #     print("Brak dostępnych punktów startowych.")
        #     return

        # spawn_transform = random.choice(spawn_points)
        spawn_transform = spawn_points[48]
        # spawn_transform.location.z += 1.0  # unieś trochę, by uniknąć kolizji z ziemią

        self.walker = self.world.try_spawn_actor(walker_bp, spawn_transform)
        if self.walker is None:
            print("Nie udało się zespawnować pieszego.")
        else:
            self.walker.set_simulate_physics(True)
            self.actor_list.append(self.walker)  # ✅ dodaj do listy
            self.walker_controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=self.walker)
        if self.walker_controller is not None:
            self.actor_list.append(self.walker_controller)  # ✅ też dodaj

        
        

        self.walker_controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=self.walker)
        self.walker_controller.start()
        self.world.tick()

        if self.walker_controller.is_alive:
            print("stworzony ale nie wie gdzie isc")
            # destination = self.world.get_random_location_from_navigation()
            # if destination is not None:
            #     self.walker_controller.go_to_location(destination)
            #     self.walker_controller.set_max_speed(random.uniform(0.5, 1.5))
            # else:
            #     print("Nie znaleziono celu dla pieszego.")

    @staticmethod
    def _calculate_distance_transform(current_location, goal_location):
        """
        Calculate distance between two locations in the environment based on the coordinates
        :param current_location: Carla waypoint class
        :param goal_location:   Carla waypoint class
        :return: distance   float
        """

        distance = math.sqrt((goal_location.location.x - current_location.location.x) ** 2 +
                             (goal_location.location.y - current_location.location.y) ** 2 +
                             (goal_location.location.z - current_location.location.z) ** 2)

        return distance

    @staticmethod
    def _calculate_distance_locations(current_location, goal_location):
        """
        Calculate distance between two locations in the environment based on the coordinates
        :param current_location: Carla Location class
        :param goal_location:   Carla Location class
        :return: distance   float
        """

        distance = math.sqrt((goal_location.x - current_location.x) ** 2 +
                             (goal_location.y - current_location.y) ** 2 +
                             (goal_location.z - current_location.z) ** 2)

        return distance

    def middle_points(self, middle_points_list):
        """

        :param middle_points_list: list of [m1: Carla transform, m2: Carla transform,
                factor: int - the distnace between middle_point1 and middle_point2 / density]
        """

        for middle_point in middle_points_list:
            mp_list = [middle_point[0], middle_point[1]]
            factor = middle_point[2]
            if factor > 0:
                self.calculate_middle_points(mp_list, factor)

    def calculate_middle_points(self, mp_list, factor):
        """
        Calculate a list of middle points between two points and append it to self.middle_points
        :param mp_list: list of 2 points of Carla transform
        :param factor: int
        """
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
                # We are not finished
                return self.calculate_middle_points(mp_list, factor)
            else:
                # Add middle point to the self.middle points
                """
                  Z coordinates can be wrong (points under or above the map)
                  Let's fix that by compering those values with route from carla method
                """
                for mp in new_mp:
                    for r in self.route:
                        r = r[0].transform.location

                        x_diff = abs(mp.location.x - r.x)
                        y_diff = abs(mp.location.y - r.y)
                        diff_sum = x_diff + y_diff

                        # If those locations are very close to each other we should set z coords to that from route
                        if diff_sum < 1:
                            # change z coords
                            mp.location.z = r.z

                # This new mid points we should place in right indexes of self.middle_goals
                for mp in new_mp:
                    index = self.middle_goals.index(mp_list[-1])
                    self.middle_goals.insert(index, mp)

    def calculate_distance(self):
        """
        Creates a mark after the car's movement (Red X)
        return: distance between two point and location of th vehicle
        """
        vehicle_location = self.vehicle.get_location()
        distance = math.sqrt((self.goal_location_loc.x - vehicle_location.x) ** 2 +
                             (self.goal_location_loc.y - vehicle_location.y) ** 2 +
                             (self.goal_location_loc.z - vehicle_location.z) ** 2)

        self.world.debug.draw_string(vehicle_location, "X", life_time=100, persistent_lines=True)
        return distance, vehicle_location

    def calculate_speed(self):
        """
        Calculate car's speed
        :return: car's speed
        """
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        return speed

    def car_control_continuous(self, action):
        """
        Manages the basic movement of a vehicle using typical driving controls.
        Instance variables:
        throttle (float) - A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
        steer (float) - A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
        brake (float) - A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0.
        hand_brake (bool) - Determines whether hand brake will be used. Default is False.
        reverse (bool) - Determines whether the vehicle will move backwards. Default is False.
        manual_gear_shift (bool) -Determines whether the vehicle will be controlled by changing gears manually. Default is False.
        gear (int) - States which gear is the vehicle running on.
        """
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
        self.control.throttle = ac.ACTION_CONTROL[self.action_space[action]][0]
        self.control.brake = ac.ACTION_CONTROL[self.action_space[action]][1]
        self.control.steer = ac.ACTION_CONTROL[self.action_space[action]][2]
        self.control.hand_brake = False
        self.control.reverse = False
        self.control.manual_gear_shift = False
        self.vehicle.apply_control(self.control)
        # i = 4 #just for debuggging

    def calculate_route_distance(self, current_location):
        # route_distance - distance to the blue route
        route_distance = min([self._calculate_distance_locations(current_location, x[0].transform.location)
                              for x in self.route])

        return route_distance

    def static_reward_mp(self, vehicle_location, static_reward_from_mp):
        """
        Calculates the distance of the vehicle to the middle points.
        :param vehicle_location:
        :return:
        """
        # Was terminal state obtained?
        done = False

        mp_distances = [self._calculate_distance_locations(vehicle_location, x[0]) for x in self.stat_reward_mp]
        mp_min = min(mp_distances)
        mp_index = mp_distances.index(mp_min)

        
        # If we are close to middle point and the reward was not already obtained
        if mp_min < 3 and self.stat_reward_mp[mp_index][1] == 0:
            self.stat_reward_mp[mp_index][1] = 1
            # If we arrive to the terminal point
            if mp_index == len(self.stat_reward_mp) - 1:
                #                             done = True
                return static_reward_from_mp, True
            else:
                return static_reward_from_mp, done
        else:
            return 0, done
    


    def reload_world(self):
        """
        Rest variables at the end of each episode
        """
        self.destroy_agents()
        self.actor_list = []


        old_world = self.client.get_world()
        if old_world is not None:
            prev_world_id = old_world.id
            del old_world
        else:
            prev_world_id = None
        

        self.world = self.client.reload_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.1
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.world.apply_settings(self.settings)

        # self.world = self.client.reload_world()



        # # SET SYNCHRONOUS MODE
        # self.settings = self.world.get_settings()
        # self.settings.synchronous_mode = True
        # self.settings.fixed_delta_seconds = 0.1
        # self.settings.max_substep_delta_time = 0.01
        # self.settings.max_substeps = 10
        # self.world.apply_settings(self.settings)
        # self.client.reload_world(False)

        tries = 3
        self.world = self.client.get_world()

        spawn_points = self.world.get_map().get_spawn_points()
        for i, spawn_point in enumerate(spawn_points):
            location = spawn_point.location
            self.world.debug.draw_string(location, str(i), draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=120.0)

        self.world.tick()
        while prev_world_id == self.world.id and tries > 0:
            tries -= 1
            self.world.tick()
            self.world = self.client.get_world()
        # self.world = self.client.reload_world()

        self.collision_history_list = []
        # History of crossing a lane markings
        self.invasion_history_list = []
        self.middle_goals = []

        # The number of steps in one episode
        self.step_counter = 0
        # Static reward from middle points
        self.stat_reward_mp = []

        # Cameras
        self.front_camera = None
        self.preview_camera = None
        self.preview_camera_enabled = False
        self.is_junction = False
        self.done = False
        self.speed = 0

    def reset(self, episode, save_image=False):
        """
        Rest environment at the end of each episode
        :return:
        """
        self.reload_world()
    
        
        if self.scenario:
            self.scenario = random.choice(self.scenario_list)
            self.create_scenario(self.sp, self.tp, self.middle_goals_density)

        # self.spawn_single_pedestrian()
        # self.spawn_npc_vehicle(spawn_index=48)

        self.plan_the_route()
        self.spawn_car(spawning_type, episode)
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
        # time.sleep(2) # added PC
        # time.sleep(0.5)

        # while self.front_camera is None:
        #     time.sleep(0.01)

        # self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        # PC
        # results_queue.put(1)
        # # A frame from the spawn point
        # while not self.image_queue.empty():
        #     _ = self.image_queue.get()

        # to render properly path on the road. Otherwise it doesn't shine
        self.step_apply_action(0)
        for i in range(15):
            self.world.tick()
            
        while not self.image_queue.empty():
            _ = self.image_queue.get()

        self.world.tick()
        try:
            image = self.image_queue.get()
        except queue.Empty:
            print("kolejka jest pusta")
            image = None

        self.state_observer.image = image

        scalar_tensor = torch.tensor(self.speed, dtype=torch.float32).view(1, 1)  # Dodanie wymiaru [batch_size=1, 1]
        batch_size = 1
        scalar_tensor = scalar_tensor.expand(batch_size, -1)  # Powielenie scalaru na cały batch
        speed_tensor = scalar_tensor.to("cuda")
        # if save_image:
        #     self.state_observer.save_to_disk(image, episode, 0)
        
        # self.process_rgb_img(image)
        self.process_semantic_img(image)
        return self.front_camera, speed_tensor
    
    def step_apply_action(self, action):
        self.step_counter += 1

        if self.action_space == 'continuous':
            self.car_control_continuous(action)
        else:
            self.car_control_discrete(action)

    def step(self, episode, step, save_image=False):
        """
        Method which creates an episode as a set of steps
        :param action: car's action
        :return:
        """
        # self.step_counter += 1

        # if self.action_space == 'continuous':
        #     self.car_control_continuous(action)
        # else:
        #     self.car_control_discrete(action)

        
        #COMMENT OUT if SYNCHROUNOUS MODE
        # if sleep_time:
        #     # How many actions per sec?
        #     time.sleep(sleep_time)

        distance_from_goal, vehicle_location = self.calculate_distance()

        route_distance = self.calculate_route_distance(vehicle_location)
        self.speed = self.calculate_speed()
        scalar_tensor = torch.tensor(self.speed, dtype=torch.float32).view(1, 1)  # Dodanie wymiaru [batch_size=1, 1]
        batch_size = 1
        scalar_tensor = scalar_tensor.expand(batch_size, -1)  # Powielenie scalaru na cały batch
        speed_tensor = scalar_tensor.to("cuda")  # Przenieś tensor na GPU

        static_reward_from_mp = mp_reward
        mp_static_reward, self.done = self.static_reward_mp(vehicle_location, static_reward_from_mp)
                
        # Was terminal state obtained?
        if self.done:
            terminal_state_reward = tp_reward
        else:
            terminal_state_reward = 0

        # How many invasions were there?
        if len(self.invasion_history_list) != 0:
            invasion_counter = 1  # Sometimes there were a lot of invasions in history only from one invasion
        else:
            invasion_counter = 0

        # Reset the history
        self.invasion_history_list = []

        reward, done = reward_function(self.collision_history_list, invasion_counter, self.speed, route_distance,
                                             mp_static_reward, terminal_state_reward)

        # Terminal state obtained or collision
        self.done = self.done or done

        if self.step_counter >= how_many_steps:
            self.done = True
        try:
            image1 = self.image_queue.get()
        except queue.Empty:
            print("Timeout: brak obrazu w kolejce")
            image1 = None
        try:
            image = self.image_queue.get() #2 frames are put on the queue between two consecutive steps
        except queue.Empty:
            print("Timeout: brak obrazu w kolejce")
            image = None
        self.state_observer.image = image

        # if save_image:
        #     self.state_observer.save_to_disk(image, episode, step)

        # self.process_rgb_img(image)
        self.process_semantic_img(image)
        return self.front_camera, reward, self.done, route_distance, speed_tensor, distance_from_goal

    def destroy_agents(self):
        """
        destroy each agent
        """
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()
        


