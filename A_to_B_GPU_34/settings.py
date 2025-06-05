import torch
SHOULD_USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global params
PORT = 2000  # Port on which the server is running
ACTION_TYPE = 'discrete'  # 'discrete' or 'continuous'
CAMERA_TYPE = 'semantic'  # 'rgb' or 'semantic'
GAMMA = 0.9  # Discount factor
LR = 1e-4  # Learning rate
USE_ENTROPY = True  # Entropy is a measure of chaos in a system
# Using the measure of entropy it is easier to avoid getting stuck in a local optima
# The name of the model which you want to load
LOAD_MODEL = 'A_to_B_GPU_34/final_models/a-b_sc7_rgb_discrete_gamma-0.9_lr-0.0001.pth'
LOAD_MODEL_PC = 'A_to_B_GPU_34/final_models/PC_MODEL'
# If you do not want to load anything keep it empty
STEP_COUNTER = 200  # How many steps in one episode?
SLEEP_BETWEEN_ACTIONS = 0.2  # How many sec sleep between consecutive actions? E.g: 0.2 gives 5 actions per 1 sec
# Without sleeping there are a lot of actions and those actions do not have enough time to influence the world.
SHOW_CAM = False  # Vehicle's camera preview
SERV_RESX = 640  # Server X resolution
SERV_RESY = 480  # Server Y resolution
SCENARIO = [8]  # List of scenarios on which the model is trained
TESTING = True
SPAWNING_TYPE = 1 # 0 - random, 1 - at the beginning, 2 - in 2 different places one by one
LOGGING = False
DRAW = False

""" 
Specify scenario parameter or spawn_point and terminal_point parameters
Scenario parameter: {1,2,3,4,5,6,7}
1) Straight short lane
2) Straight long lane
3) Turn right
4) Turn left
5) Little straight lane and right turn
6) Little straight lane and left turn
7) Straight lane + 2 right turns
8) Straight + Left + Straight + Right + Straight
10) Random paths, based on set of carla points defined in carla_env.py
11) Random choice between different sets of points, defined for specific manouvers (different turns on different junctions)


The whole list of scenarios can be specified with switch_scenario parameter indicating that
after switch_scenario episodes there will be next scenario from the list
"""

# Rewards
REWARD_FROM_TP = 0  # Static reward from arriving to the terminal point
REWARD_FROM_MP = 0  # Static reward from arriving to the middle point
REWARD_FROM_COL = 0  # Static reward from a collision
REWARD_FROM_INV = 0  # Static reward from a line invasion
# Speed and distance rewards are in the utils.py/reward_function

# ACTIONS = ['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']

ACTIONS = ['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right', 'forward_slight_left','forward_slight_right','brake_slight_left','brake_slight_right',]
# ACTION_IND:  0            1               2             3          4              5                 6                    7                       8                    9

CARLA_PATH = r''  # provide your carla exec path
CARLA_EGG_PATH = r''  # Provide your carla egg path

