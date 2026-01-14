import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Add the environment
# from environment import 

models_dir = f"models/PPO-{int(time.time())}"
logs_dir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Setup the environment
# env = 
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = 25000
for i in range(1, 40):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_run{i}")
    model.save(f"{models_dir}/PPO_pilotGame_{TIMESTEPS*i}")

env.close()