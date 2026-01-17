#!/usr/bin/env python3
"""
GAIT Assignment 3 - Part 2: Deep RL real time Arena training
"""

import sys
import pygame
from typing import Optional, Tuple
from environment import SPEEN_AND_VROOM, BORING_4D_PAD, SPEEN_VROOM_ALL_ACTIONS, BORING_4D_PAD_ALL_ACTIONS
from environment.arena import ArenaEnv, A_NONE, A_SHOOT, A_1_FORWARD, A_1_LEFT, A_1_RIGHT, A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT, ALL_ACTIONS
from environment import entities


# For Debugging
arena = ArenaEnv(BORING_4D_PAD, render_mode="human")
if arena is None:
    print("environment fail")
else:
    print("environment success")


clock = pygame.time.Clock()

running = True
forward = False
upward = False
leftward = False
rightward = False
downward = False
e = None
arena.reset()
while running:
    control = 0
    do = A_NONE
    control_style = None
    
    for event in pygame.event.get():
        if event == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_w:
                upward = True
            elif event.key == pygame.K_a:
                leftward = True
            elif event.key == pygame.K_d:
                rightward = True
            elif event.key == pygame.K_s:
                downward = True
            if event.key == pygame.K_UP:
                forward = True
            elif event.key == pygame.K_LEFT:
                control_style = SPEEN_AND_VROOM
                do = A_1_LEFT
            elif event.key == pygame.K_RIGHT:
                control_style = SPEEN_AND_VROOM
                do = A_1_RIGHT
            elif event.key == pygame.K_z or event.key == pygame.K_j:
                do = A_SHOOT
            if event.key == pygame.K_SPACE:
                for htb in arena.hittables[:]:
                    if not (isinstance(htb, entities.Agent) or isinstance(htb, entities.Player)):
                        htb.destroy()
            if event.key == pygame.K_p:
                for htb in arena.hittables[:]:
                    if isinstance(htb, entities.Enemy):
                        htb.destroy()
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                upward = False
            elif event.key == pygame.K_a:
                leftward = False
            elif event.key == pygame.K_d:
                rightward = False
            elif event.key == pygame.K_s:
                downward = False
            if event.key == pygame.K_UP:
                forward = False
    
    if forward and do != A_SHOOT:
        control += 1
        control_style = SPEEN_AND_VROOM
        do = A_1_FORWARD
    if control_style is None and do != A_SHOOT:
        if upward:
            control += 1
            control_style = BORING_4D_PAD
            do = A_2_UP
        if downward:
            control += 1
            control_style = BORING_4D_PAD
            do = A_2_DOWN
        if leftward:
            control += 1
            control_style = BORING_4D_PAD
            do = A_2_LEFT
        if rightward:
            control += 1
            control_style = BORING_4D_PAD
            do = A_2_RIGHT
        if control > 1:
            do = A_NONE
    arena.step(ALL_ACTIONS[arena.control_style].index(do))
    arena.render(0, 0, "Debug: SPACE clear all Enemy and Spawner")
    if not arena.alive:
        running = False
    clock.tick(60)