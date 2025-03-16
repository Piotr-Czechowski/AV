import wandb
import numpy as np
from collections import Counter
import plotly.express as px
import pandas as pd

""" Code shows histogram of in how many episodes finished with specific value of reward"""

# Initialize the W&B API
api = wandb.Api(timeout=90)

# specify credentials
entity = "bartekkawa2021-agh-university-of-science-and-technology"  # W&B user name
project = "A_to_B"  # project name
run_id = "synchr_200_semantic_camera_6_13_img_speed_2"  # run name

run = api.run(f"{entity}/{project}/{run_id}")

# collect data
history = run.history(samples=2000, keys=['reward'])

# reward per episode data
reward_per_episode = []
for idx, val in enumerate(history['reward']):
    if val is not None:
        if not np.isnan(val):
            print(f'idx {idx} and val {val}')
            reward_per_episode.append(val)

# TODO select correct range if necessary
# low_limit, up_limit = 20, 600
# reward_per_episode = [reward for reward in reward_per_episode if reward > low_limit and reward < up_limit]

# round and sort
rounded_rewards = Counter([round(reward) for reward in reward_per_episode])
sorted_counter = dict(sorted(rounded_rewards.items()))

# plot results
df = pd.DataFrame({
    "Category": list(sorted_counter.keys()),
    "Values": list(sorted_counter.values())
})
# Create the bar chart with hover data
fig = px.bar(df, x="Category", y="Values", color="Values",
             hover_data={"Values": True},
             color_continuous_scale="plasma")
# Show figure
fig.show()
