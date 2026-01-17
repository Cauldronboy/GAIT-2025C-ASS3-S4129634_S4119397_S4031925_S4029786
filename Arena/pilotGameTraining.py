import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import time

# Add the environment
from environment import ArenaEnv, SPEEN_AND_VROOM, BORING_4D_PAD

import torch
torch.set_num_threads(8)

if __name__ == "__main__":
    def make_env(control_style):
        def _init():
            return Monitor(ArenaEnv(control_style=control_style))
        return _init

    num_envs = 4

    models1_dir = f"models/models_control_style_1/PPO"
    logs1_dir = f"logs/logs_control_style_1/PPO"

    if not os.path.exists(models1_dir):
        os.makedirs(models1_dir)
    if not os.path.exists(logs1_dir):
        os.makedirs(logs1_dir)

    # Setup the environment
    env1 = SubprocVecEnv([make_env(SPEEN_AND_VROOM) for _ in range(num_envs)])

    agent1 = PPO(
        "MlpPolicy",
        env1,
        n_steps=1024,        # IMPORTANT with VecEnv
        batch_size=1024,
        n_epochs=5,
        verbose=1,
        tensorboard_log=logs1_dir
    )

    TIMESTEPS = 25000 // num_envs
    for i in range(1, 40):
        agent1.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        agent1.save(f"{models1_dir}/PPO_pilotGame_{TIMESTEPS*i}")

    print("Saving final model for agent 1...")
    agent1.save(f"{models1_dir}/PPO_final")
    env1.close()

    # Second training agent (Different action space)
    models2_dir = f"models/models_control_style_2/PPO"
    logs2_dir = f"logs/logs_control_style_2/PPO"

    if not os.path.exists(models2_dir):
        os.makedirs(models2_dir)
    if not os.path.exists(logs2_dir):
        os.makedirs(logs2_dir)

    # Setup the 2nd environment
    env2 = SubprocVecEnv([make_env(BORING_4D_PAD) for _ in range(num_envs)])

    agent2 = PPO(
        "MlpPolicy",
        env2,
        n_steps=1024,
        batch_size=1024,
        n_epochs=5,
        verbose=1,
        tensorboard_log=logs2_dir
    )

    TIMESTEPS = 25000 // num_envs
    for i in range(1, 40):
        agent2.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        agent2.save(f"{models2_dir}/PPO_pilotGame_{TIMESTEPS*i}")

    print("Saving final model for agent 2...")
    agent2.save(f"{models2_dir}/PPO_final")

    env2.close()