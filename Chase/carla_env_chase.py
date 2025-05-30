import glob
import os
import queue
import random
import sys
import time
import math
import torch
import pickle
# import pygame
# import threading

import settings
from utils import ColoredPrint, reward_function, Timer
from ACTIONS import ACTIONS as ac
from state_observer import StateObserver

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

# Global settings
fps = settings.FPS
show_cam = settings.SHOW_CAM
sleep_time = settings.SLEEP_BETWEEN_ACTIONS
serv_resx = settings.SERV_RESX
serv_resy = settings.SERV_RESY


def start_carla_server(args):
    """Start carla server"""
    if os.name == 'nt':  # Windows
        return subprocess.Popen(f'CarlaUE4.exe ' + args, cwd=settings.CARLA_PATH, shell=True)
    elif os.name == 'posix':  # Ubuntu
        # return os.system(f'{settings.CARLA_PATH}/CarlaUE4.sh ' + args + ' &')
        return subprocess.Popen(f'{settings.CARLA_PATH}/CarlaUE4.sh ' + args, cwd=settings.CARLA_PATH, shell=True)


class CarlaEnv:
    """Create Carla environment"""
    def __init__(self, scenario, action_space='discrete',  camera='rgb', res_x=80, res_y=80, port=2000,
                 manual_control=False):
        # Run the server on 127.0.0.1/port
        # start_carla_server(f'-windowed -carla-server -carla-rpc-port={port} -ResX={serv_resx} -ResY={serv_resy} '
                        #    f'-quality-level=Low -fps={fps}')
        # -carla-port
        # -carla-world-port
        # -carla-rpc-port
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(30.0)

        # Make sure that server and client versions are the same
        client_ver = self.client.get_client_version()
        server_ver = self.client.get_server_version()

        self.log = ColoredPrint()  # Enable to use colors
        if client_ver == server_ver:
            self.log.success(f"Client version: {client_ver}, Server version: {server_ver}")
        else:
            self.log.warn(f"Client version: {client_ver}, Server version: {server_ver}")

        self.world = self.client.load_world('Town03')
        self.settings = self.world.get_settings()
        self.camera_type = camera
        self.blueprint_library = self.world.get_blueprint_library()  # List of available agents with attributes
        self.map = self.world.get_map()

        self.scenario_list = scenario
        try:
            self.scenario = self.scenario_list[0]  # Single scenario
        except IndexError:
            self.scenario = False
        """
        a - vehicle which is chasing
        b - vehicle which is being chased
        sp - spawn point
        """
        self.a_sp, self.b_sp, self.ride_history = self.create_scenario()

        # self.history_frame_number = 0
        self.a_sp_loc, self.b_sp_loc = self.a_sp.location, self.b_sp.location
        

        self.set_spectator()  # Set the spectator
        self.action_space = self.create_action_space(action_space)  # Environment possible actions
        self.actor_list = []  # List of all actors in the environment
        self.transform = carla.Transform(carla.Location(x=2.5, z=0.7))  # Location of attached sensors to the vehicle
        self.manual_control = manual_control

        # self.ride_iterator = None
        self.b_vehicle = self.spawn_car('carlacola', self.b_sp)  # Chased car
        self.prev_b_vehicle_loc = 0  # Previous location of the chased car. Used for drawing a route
        # if not manual_control:
        self.a_vehicle = self.spawn_car('model3', self.a_sp, ego=True)  # Car which is chasing

        # Manages the basic movement of a vehicle using typical driving controls
        self.control = carla.VehicleControl()

        # Images X, Y resolutions
        self.res_x = res_x
        self.res_y = res_y

        # Cameras
        self.show_cam = show_cam
        self.front_camera = None

        # self.clock = pygame.time.Clock()  # Controls FPS of pygame client
        self.collision_history_list = []  # Variables which have to reset at the end of each episode
        self.step_counter = 0  # The number of steps in one episode
        self.thread = None  # Vehicle B movement thread
        self.done = False  # A flag which ends the episode
        # self.episode_timer = Timer()  # Episode's timer
        # self.effective_chase_timer = Timer()  # Measures the time of being from 5m to 25m around the chased car
        self.effective_chase_per = 0  # Effective chase / whole duration * 100%
        self.ride_history = []

        self.state_observer = StateObserver()
        self.previous_step_distance = 100

    def create_scenario(self):
        """
        Create a scenario based on integer, input value
        """
        if self.scenario == 1:
            # Short straight
            ride_file = 'Chase/drives/ride1.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map
            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x -= 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)

            # for i in range(len(ride_history) - 300):  # Make the ride shorter
            #     ride_history.pop()

            return a_sp, b_sp, ride_history

        elif self.scenario == 2:
            # Long straight
            ride_file = 'Chase/drives/ride1.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map
            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x -= 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)

            return a_sp, b_sp, ride_history

        elif self.scenario == 3:
            # Turn left
            ride_file = 'Chase/drives/ride7.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map

            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x += 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)
            for i in range(len(ride_history) - 250):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        elif self.scenario == 4:
            # Slow turn left
            ride_file = 'Chase/drives/ride9.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map

            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x += 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)
            for i in range(len(ride_history) - 300):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        elif self.scenario == 5:
            # Turn right
            ride_file = 'Chase/drives/ride14.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map

            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x += 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)
            for i in range(len(ride_history) - 350):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        else:
            self.log.err(f"Invalid params: scenario: {self.scenario}")

    @staticmethod
    def load_ride(filepath):
        """
        Load the history of a ride
        :param filepath: pickle file
        :return: full ride hisotry and a spawn point for a chased vehicle
        """
        # ride_history = pickle.load(open(filepath, 'rb'))
        with open(filepath, 'rb') as file:  # Konstrukcja "with" zapewnia zamknięcie pliku
            ride_history = pickle.load(file)

        sp = ride_history[0]
        b_sp_loc = carla.Location(sp[0], sp[1], sp[2])
        b_sp_rot = carla.Rotation(sp[3], sp[4], sp[5])

        return ride_history, b_sp_loc, b_sp_rot

    # def set_spectator(self):
    def set_spectator(self, d=6.4):

        """
        Get specator's camera angles
        :param d: constant
        :return: self.spectator - spectator's exact location and its angles
        """
        spectator_coordinates = carla.Location(self.a_sp_loc.x,
                                               self.a_sp_loc.y,
                                               self.a_sp_loc.z)
        rotation = carla.Rotation(0, 0, 0)

        if self.scenario in [1, 2]:
            spectator_coordinates.x -= 3
            spectator_coordinates.z += 30
            rotation = carla.Rotation(yaw=0, pitch=-50, roll=0)

        elif self.scenario in [3, 4]:
            spectator_coordinates.x -= 3
            spectator_coordinates.z += 45
            rotation = carla.Rotation(yaw=-210, pitch=-70, roll=0)

        elif self.scenario == 5:
            spectator_coordinates.x -= 3
            spectator_coordinates.z += 45
            rotation = carla.Rotation(yaw=-140, pitch=-70, roll=0)

        self.spectator = self.world.get_spectator()
        """
        yaw - rotating your vision in 2D (left <-, right ->)
        pitch - looking more to the sky or the road 
        roll - leaning your vision (e.g. from | to ->)
        """
        self.spectator.set_transform(carla.Transform(spectator_coordinates, rotation))
        
        return self.spectator

    def spawn_car(self, model_name, spawn_point, ego=False):
        """
        Spawn a car
        :return: vehicle
        """
        bp = self.blueprint_library.filter(model_name)[0]
        if ego:
            bp.set_attribute('role_name', 'ego')
        vehicle = self.world.try_spawn_actor(bp, spawn_point)
        self.actor_list.append(vehicle)
        return vehicle

    def create_action_space(self, action_space):
        """
        Create an action space for an agent
        :param action_space: discrete or continuous
        :return: possible actions to take for an agent
        """
        if action_space == 'discrete':
            self.action_space = [getattr(ac, action) for action in settings.ACTIONS]
            return self.action_space
        else:  # Continuous
            self.action_space = action_space
            return self.action_space

    def add_rgb_camera(self, vehicle):
        """
        Attach RGB camera to the vehicle
        The "RGB" camera acts as a regular camera capturing images from the scene
        """
        rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{self.res_x}")
        rgb_cam_bp.set_attribute("image_size_y", f"{self.res_y}")
        rgb_cam_bp.set_attribute("fov", f"110")

        rgb_cam = self.world.spawn_actor(rgb_cam_bp, self.transform, attach_to=vehicle)
        self.actor_list.append(rgb_cam)
        self.image_queue = queue.Queue()
        rgb_cam.listen(self.image_queue.put)

    def process_rgb_img(self, image):
        """
        Process RGB images
        :param image: raw data from the rgb camera
        :return:
        """
        i = np.array(image.raw_data)
        # Also returns alfa values - not only rgb
        i2 = i.reshape((self.res_y, self.res_x, 4))
        i3 = i2[:, :, :3]

        if self.show_cam:
            # noinspection PyUnresolvedReferences
            cv2.imshow("", i3)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

        self.front_camera = torch.Tensor(i3).view(3, self.res_x, self.res_y).unsqueeze(0)

    def add_semantic_camera(self, vehicle):
        """
        Attach semantic camera to the vehicle
        The "Semantic Segmentation" camera classifies every object in the view by displaying it in a different color
        according to the object class. E.g. pedestrians appear in a different color than vehicles.
        Original images are totally black
        """
        semantic_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_cam_bp.set_attribute('image_size_x', f'{self.res_x}')
        semantic_cam_bp.set_attribute('image_size_y', f'{self.res_y}')
        semantic_cam_bp.set_attribute('fov', '110')

        semantic_cam_sensor = self.world.spawn_actor(semantic_cam_bp, self.transform, attach_to=vehicle)

        semantic_cam_sensor.listen(lambda data: self.process_semantic_img(data))
        self.actor_list.append(semantic_cam_sensor)

    def process_semantic_img(self, image):
        """
        Process semantic images
        :param image: raw data from the semantic camera
        """
        image.convert(cc.CityScapesPalette)
        image = np.array(image.raw_data)
        image = image.reshape((self.res_x, self.res_y, 4))
        image = image[:, :, :3]
        if self.show_cam:
            # noinspection PyUnresolvedReferences
            cv2.imshow("", image)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

        self.front_camera = torch.Tensor(image).view(3, self.res_x, self.res_y).unsqueeze(0).float()

    def add_collision_sensor(self, vehicle):
        """
        This sensor, when attached to an actor, it registers an event each time the actor collisions against sth
        in the world. This sensor does not have any configurable attribute.
        """
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        col_sensor_bp = self.world.spawn_actor(col_sensor_bp, self.transform, attach_to=vehicle)
        col_sensor_bp.listen(lambda data: self.collision_data_registering(data))
        self.actor_list.append(col_sensor_bp)

    def collision_data_registering(self, event):
        """
        Register collisions
        :param event: data from the collision sensor
        """
        coll_type = event.other_actor.type_id
        self.collision_history_list.append(event)

    def car_control_continuous(self, action, vehicle):
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
        self.a_vehicle.apply_control(self.control)

    def car_control_discrete(self, action):
        self.control.throttle = ac.ACTION_CONTROL[self.action_space[action]][0]
        self.control.brake = ac.ACTION_CONTROL[self.action_space[action]][1]
        self.control.steer = ac.ACTION_CONTROL[self.action_space[action]][2]
        self.control.hand_brake = False
        self.control.reverse = False
        self.control.manual_gear_shift = False
        self.a_vehicle.apply_control(self.control)

    @staticmethod
    def calculate_distance(a_location, b_location):
        """
        Calculate distance between two locations in the environment based on the coordinates
        :param a_location:   Carla Location class
        :param b_location:   Carla Location class
        :return: distance    float
        """

        distance = math.sqrt((b_location.x - a_location.x) ** 2 +
                             (b_location.y - a_location.y) ** 2 +
                             (b_location.z - a_location.z) ** 2)

        return distance

    @staticmethod
    def calculate_angle(a_vehicle, b_vehicle):
        """
        Calculate the angle between two actors
        :param a_vehicle: (carla.Actor)
        :param b_vehicle: (carla.Actor)
        :return: angle between two actors
        """
        a_rotation = a_vehicle.get_transform().rotation
        b_rotation = b_vehicle.get_transform().rotation
        angle = a_rotation.yaw - b_rotation.yaw

        return abs(angle)

    def draw_movement(self, vehicle):
        """
        Creates a mark after the car's movement (green X)
        :param vehicle: (carla.Actor)
        return: location of the vehicle
        """
        vehicle_location = vehicle.get_location()
        green = carla.Color(0, 255, 0)
        red = carla.Color(255, 0, 0)

        if vehicle.type_id == "vehicle.tesla.model3":  # Chasing car
            self.world.debug.draw_string(location=vehicle_location, text="X", color=green, life_time=100)
        elif self.prev_b_vehicle_loc:  # Chased car. if it is not the first step
            self.world.debug.draw_line(begin=self.prev_b_vehicle_loc, end=vehicle_location, thickness=0.3, color=red,
                                       life_time=100)

        return vehicle_location

    def chased_vehicle_movement(self):
        """
        Teleports a chased vehicle to the next location from the file
        """
        # for frame in self.ride_history:
        #     if not self.done:
        #         if self.scenario == 5:
        #             time.sleep(0.03)
        #         else:
        #             time.sleep(0.07)
        #         self.draw_movement(self.b_vehicle)
        #         self.prev_b_vehicle_loc = self.b_vehicle.get_transform().location
        #         new_loc = carla.Location(frame[0], frame[1], frame[2])
        #         new_rotation = carla.Rotation(frame[3], frame[4], frame[5])

        #         new_point = carla.Transform(new_loc, new_rotation)
        #         self.b_vehicle.set_transform(new_point)

    def reload_world(self):
        """
        Rest Carla env and variables at the end of each episode
        """
        self.destroy_agents()
        self.actor_list = []


        old_world = self.client.get_world()
        if old_world is not None:
            prev_world_id = old_world.id
            del old_world
        else:
            prev_world_id = None

        # if not self.manual_control:
        self.world=self.client.reload_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.1
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.world.apply_settings(self.settings)

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

        self.collision_history_list = []
        self.prev_b_vehicle_loc = None
        self.step_counter = 0
        self.done = False
        self.front_camera = None
        # self.episode_timer = Timer()
        # self.effective_chase_timer = Timer()
        self.effective_chase_per = 0
        # self.ride_history.clear()
        self.history_frame_number = 0
        self.previous_step_distance = 100
        self.a_previous_location = None
        self.b_previous_location = None


    def reset(self, vehicle_for_mc=None):
        """
        Rest environment at the end of each episode
        :return: self.front_camera - an 80x80 image from the spawn point
        """
        # if self.manual_control:
        #     self.a_vehicle = vehicle_for_mc

        # if self.step_counter > 0:  # Omit the first iteration
        #     # self.thread.join()
        #     pass

        self.reload_world()  # Reset variables
        self.scenario = random.choice(self.scenario_list)  # Get random scenario
        self.a_sp, self.b_sp, self.ride_history = self.create_scenario()
        # self.b_sp = self.map.get_spawn_points()[3]

        self.b_vehicle = self.spawn_car('carlacola', self.b_sp)  # Chased car
        self.a_vehicle = self.spawn_car('model3', self.a_sp, ego=True)  # Car which is chasing

        # self.ride_iterator = iter(self.ride_history)
        self.no_more_b_vehicle_points = False
        self.b_vehicle_frame = self.ride_history[0]
        
        new_loc = carla.Location(self.b_vehicle_frame[0], self.b_vehicle_frame[1], self.b_vehicle_frame[2])
        new_rotation = carla.Rotation(self.b_vehicle_frame[3], self.b_vehicle_frame[4], self.b_vehicle_frame[5])

        new_point = carla.Transform(new_loc, new_rotation)
        self.b_vehicle.set_transform(new_point)
        self.world.tick()
        self.prev_b_vehicle_loc = self.b_vehicle.get_transform().location


        
        self.a_sp_loc, self.b_sp_loc = self.a_sp.location, self.b_sp.location
        self.spectator = self.set_spectator() # Set the spectator

        # self.b_vehicle = self.spawn_car('carlacola', self.b_sp)  # Chased car
        # self.a_vehicle = self.spawn_car('model3', self.a_sp)  # Car which is chasing

        if self.camera_type == 'rgb':
            self.add_rgb_camera(self.a_vehicle)
        elif self.camera_type == 'semantic':
            self.add_semantic_camera(self.a_vehicle)
        else:
            self.log.err(f"Wrong camera type. Pick rgb or semantic, not: {self.camera_type}")

        # self.add_collision_sensor(self.a_vehicle)
        # self.a_vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        # time.sleep(0.5)

        # while self.front_camera is None:
        #     time.sleep(0.01)
        self.world.tick()
        self.world.tick()    
        self.world.tick()
        self.world.tick()
        self.world.tick()
        self.world.tick()

        # self.a_vehicle.apply_control(carla.VehicleControl(brake=0.0))
        while not self.image_queue.empty():
            _ = self.image_queue.get()

        self.world.tick()
        image = self.image_queue.get()
        self.state_observer.image = image

        self.process_rgb_img(image)
        return self.front_camera
    
    def step_apply_action(self, action, vehicle_for_mc=None):
        self.step_counter += 1
        if self.action_space == 'continuous':
            self.car_control_continuous(action)
        else:
            self.car_control_discrete(action)
        #?????????????????
        # if not self.manual_control:#?????????????????
        #     if self.action_space == 'continuous':
        #         self.car_control_continuous(action)
        #     else:
        #         self.car_control_discrete(action)
        # else:
        #     self.a_vehicle = vehicle_for_mc

    def step(self):
        """
        # Method which creates an episode as a set of steps
        # :param action: car's action
        # :param vehicle_for_mc: carla.Vehicle class used only in human_performance_test.py
        # :return:
        # """
        # # self.clock.tick(fps)  # Pygame's FPS #??????

        # # if not self.manual_control:
        # #     if self.action_space == 'continuous':
        # #         self.car_control_continuous(action, self.a_vehicle)
        # #     else:  # Discrete
        # #         self.car_control_discrete(action, self.a_vehicle)
        # # else:
        # #     self.a_vehicle = vehicle_for_mc

        # if self.step_counter == 0:  # Only the first step
        #     # Apply chased vehicle movement async
        #     # self.thread = threading.Thread(target=self.chased_vehicle_movement, args=(), kwargs={}, daemon=True)
        #     # self.thread.start()
        #     # self.episode_timer.start()
        #     # Assumption that the chase is starting at point where the chasing car is 5-25m behind
        #     # self.effective_chase_timer.start()
        #     pass

        self.step_counter += 1

        # # if self.no_more_b_vehicle_points == False:
        # #     try:
        # #         self.b_vehicle_frame = next(self.ride_iterator)
        # #         self.b_vehicle_frame = next(self.ride_iterator)
        # #         self.draw_movement(self.b_vehicle)
        # #         self.prev_b_vehicle_loc = self.b_vehicle.get_transform().location
        # #         new_loc = carla.Location(self.b_vehicle_frame[0], self.b_vehicle_frame[1], self.b_vehicle_frame[2])
        # #         new_rotation = carla.Rotation(self.b_vehicle_frame[3], self.b_vehicle_frame[4], self.b_vehicle_frame[5])

        # #         new_point = carla.Transform(new_loc, new_rotation)
        # #         self.b_vehicle.set_transform(new_point)
        # #     except StopIteration:
        # #         self.no_more_b_vehicle_points = True
        # #         del self.ride_iterator
        # #         del self.b_vehicle_frame
        # # else:
        # #     pass
        #     # self.world.tick()
        if self.history_frame_number < len(self.ride_history):
            self.b_vehicle_frame = self.ride_history[self.history_frame_number]
            new_loc = carla.Location(self.b_vehicle_frame[0], self.b_vehicle_frame[1], self.b_vehicle_frame[2])
            new_rotation = carla.Rotation(self.b_vehicle_frame[3], self.b_vehicle_frame[4], self.b_vehicle_frame[5])

            new_point = carla.Transform(new_loc, new_rotation)
            self.b_vehicle.set_transform(new_point)
            self.history_frame_number += 3
            # self.draw_movement(self.b_vehicle) # Comment and try to chase without this shinig movement line
            self.prev_b_vehicle_loc = self.b_vehicle.get_transform().location


        a_location, b_location = self.draw_movement(self.a_vehicle), self.b_vehicle.get_location()
        ab_distance = round(self.calculate_distance(a_location, b_location), 3)
        if self.a_previous_location is None:
            self.a_previous_location = a_location
            self.b_previous_location = b_location

        apb_distance = round(self.calculate_distance(self.a_previous_location, b_location), 3) #distance between b and previous a

        # # if 5 <= ab_distance <= 25:  # 5m-25m effective chase
        # #     if self.effective_chase_timer.paused:
        # #         self.effective_chase_timer.resume()
        # # else:
        # #     if not self.effective_chase_timer.paused:
        # #         self.effective_chase_timer.pause()

        # # Done from a collision or a distnace
        reward, self.done = reward_function(self.collision_history_list, ab_distance=ab_distance, apb_distance=apb_distance, ab_previous_distance=self.previous_step_distance, a_location=a_location, b_location=b_location, a_previous_location=self.a_previous_location, b_previous_location=self.b_previous_location)
        self.previous_step_distance = ab_distance
        self.a_previous_location = a_location
        self.b_previous_location = b_location
        # reward, self.done = reward_function(self.collision_history_list, ab_distance=2,
        #                                     timer=0) # timer=self.episode_timer.get())

        # if not self.thread.is_alive():
            # self.done = True  # The end of the episode


        image1 = self.image_queue.get()
        image = self.image_queue.get() #2 frames are put on the queue between two consecutive steps
        self.state_observer.image = image

        if self.done:
            """ Calculate effective chase time"""
            # self.effective_chase_per = self.effective_chase_timer.get() / self.episode_timer.get() * 100
            print("Done")

        self.process_rgb_img(image)

        # return self.front_camera, reward, self.done
        return self.front_camera, reward, self.done

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
