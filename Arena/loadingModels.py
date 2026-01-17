import gymnasium as gym
from stable_baselines3 import PPO
import pygame

from environment import ArenaEnv, SPEEN_AND_VROOM, BORING_4D_PAD

env = ArenaEnv(control_style=SPEEN_AND_VROOM, render_mode="human")

models_dir = f"models/models_control_style_1/PPO"
models_path = f"{models_dir}/PPO_pilotGame_243750.zip" # Change to desired model path

model = PPO.load(models_path, env=env)


running = True
reward_this_ep = 0
reset_count = 0

obs, info = env.reset()

while running:
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    reward_this_ep += reward
    
    env.render(
        reset_count=reset_count,
        info=f"Reward this episode: {reward_this_ep:.2f}"
    )
    
    if terminated or truncated:
        obs, info = env.reset()
        reward_this_ep = 0
        reset_count += 1
        print("Episode reset")
    
    reward_this_ep = 0.0
    reset_count += 1

env.close()
pygame.quit()