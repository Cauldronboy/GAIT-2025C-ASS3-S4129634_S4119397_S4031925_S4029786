#!/usr/bin/env python3
"""
Test script to verify arena rendering
"""

import pygame
from environment import ArenaEnv

# Create environment with rendering enabled
env = ArenaEnv(render_mode="human")

# Reset and render
obs, info = env.reset()
print("Environment reset successfully")
print(f"Initial observation shape: {obs.shape}")

reward_this_ep = 0
reset_count = 0

# Run a few steps and render
running = True
clock = pygame.time.Clock()

while running:
    # Control frame rate
    clock.tick(60)

    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    reward_this_ep += reward

    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    
    if terminated or truncated:
        obs, info = env.reset()
        reward_this_ep = 0
        reset_count += 1
        print("Episode reset")
    
    extra_info = f"Reward this episode: {reward_this_ep}"

    # Render the current state
    env.render(reset_count=reset_count, info=extra_info)

env.close()
print("Test completed")
