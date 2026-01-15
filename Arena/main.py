#!/usr/bin/env python3
"""
GAIT Assignment 3 - Part 2: Deep RL real time Arena training
"""

import sys
import pygame
from typing import Optional, Tuple
from environment import Arena, ArenaRenderer, SPEEN_AND_VROOM, BORING_4D_PAD, A_NONE, A_SHOOT, A_1_FORWARD, A_1_LEFT, A_1_RIGHT, A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT

arena = Arena()
renderer = ArenaRenderer()
if arena is None:
    print("environment fail")
else:
    print("environment success")

renderer.init_display(arena)

clock = pygame.time.Clock()

running = True
forward = False
upward = False
leftward = False
rightward = False
downward = False
while running:
    do = A_NONE
    if forward:
        do = A_1_FORWARD
    if upward and rightward and leftward and rightward:
        do = A_NONE
    elif upward:
        do = A_2_UP
    elif downward:
        do = A_2_DOWN
    elif leftward:
        do = A_2_LEFT
    elif rightward:
        do = A_2_RIGHT
    
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
                do = A_1_LEFT
            elif event.key == pygame.K_RIGHT:
                do = A_1_RIGHT
            elif event.key == pygame.K_z or event.key == pygame.K_j:
                do = A_SHOOT
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
    arena.step(style=1, action=do)
    renderer.render(arena)
    if not arena.alive:
        running = False
    clock.tick(60)