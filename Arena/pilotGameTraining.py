import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Add the environment
from environment.arenaEnv import ArenaEnv

models1_dir = f"models_control_style_1/PPO-{int(time.time())}"
logs1_dir = f"logs_control_style_1/PPO-{int(time.time())}"

if not os.path.exists(models1_dir):
    os.makedirs(models1_dir)
if not os.path.exists(logs1_dir):
    os.makedirs(logs1_dir)

# Setup the environment
env1 = ArenaEnv()
env1.reset()

agent1 = PPO("MlpPolicy", env1, verbose=1, tensorboard_log=logs1_dir)

TIMESTEPS = 25000
for i in range(1, 40):
    agent1.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_run{i}")
    agent1.save(f"{models1_dir}/PPO_pilotGame_{TIMESTEPS*i}")

env1.close()

# Second training agent (Different action space)
models2_dir = f"models_control_style_2/PPO-{int(time.time())}"
logs2_dir = f"logs_control_style_2/PPO-{int(time.time())}"

if not os.path.exists(models2_dir):
    os.makedirs(models2_dir)
if not os.path.exists(logs2_dir):
    os.makedirs(logs2_dir)

# Setup the 2nd environment
env2 = ArenaEnv()
env2.reset()

agent2 = PPO("MlpPolicy", env2, verbose=1, tensorboard_log=logs2_dir)

TIMESTEPS = 25000
for i in range(1, 40):
    agent2.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_run{i}")
    agent2.save(f"{models2_dir}/PPO_pilotGame_{TIMESTEPS*i}")

env2.close()