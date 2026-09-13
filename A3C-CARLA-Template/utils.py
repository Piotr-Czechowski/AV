import settings
from datetime import datetime
import math


class ColoredPrint:
    def __init__(self):
        self.PINK = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'

    def disable(self):
        self.PINK = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

    def store(self):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('logfile.log', mode='a') as file_:
            file_.write(f"{self.msg} -- {date}")
            file_.write("\n")

    def success(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.OKGREEN + self.msg + self.ENDC, **kwargs)
        return self

    def info(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.OKBLUE + self.msg + self.ENDC, **kwargs)
        return self

    def warn(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.WARNING + self.msg + self.ENDC, **kwargs)
        return self

    def err(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.FAIL + self.msg + self.ENDC, **kwargs)
        return self

    def pink(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.PINK + self.msg + self.ENDC, **kwargs)
        return self


def reward_function(collision_history_list, invasion_counter, speed, route_distance, mp_static_reward,
                          terminal_state_reward, on_junction, prev_speed):

    if len(collision_history_list) != 0 or route_distance >= 10:
        done = True
        col_reward = settings.REWARD_FROM_COL
    else:
        done = False
        col_reward = 0

    inv_reward = invasion_counter * settings.REWARD_FROM_INV

    speed_reward = -1.2 + 4*math.sin(speed/10) # pik jest w okolicach 20 km/h
    if route_distance < 1.5:
        route_distance_reward = 1
        if on_junction and speed_reward > 0:
            route_distance_reward = route_distance_reward * 4
    else:
        route_distance_reward = -4*math.sin(speed/10)

    reward = terminal_state_reward + col_reward + speed_reward + route_distance_reward + inv_reward + mp_static_reward

    return reward, done
