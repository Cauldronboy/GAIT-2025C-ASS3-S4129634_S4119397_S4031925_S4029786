#!/usr/bin/env python3
"""
Test script to verify arena rendering
"""

import pygame
from Arena.environment.arena import ArenaEnv

# Create environment with rendering enabled
env = ArenaEnv(render_mode="human")

# Reset and render
obs, info = env.reset()
print("Environment reset successfully")
print(f"Initial observation shape: {obs.shape}")

# Run a few steps and render
running = True
clock = pygame.time.Clock()

while running:
    # Render the current state
    env.render()
    
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Control frame rate
    clock.tick(60)
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Episode reset")

env.close()
print("Test completed")
