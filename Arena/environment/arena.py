"""
Core Arena environment for reinforcement learning.
Polygonkind is dead
Area is fuel
GPU is full
"""

import pygame
import math
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, Optional
import vectorHelper


# Action sets
CONTROLSTYLE1 = [
    (0, 0, 0),      # (rotation angle, move forward, shoot)
    (-1, 0, 0),     # turn left
    (1, 0, 0),      # turn right
    (0, 1, 0),      # move forward
    (0, 0, 1),      # shoot
]
CONTROLSTYLE2 = [
    (0, 0, 0),      # (x movement, y movement, shoot)
    (0, -1, 0),     # move up
    (0, 1, 0),      # move down
    (-1, 0, 0),     # move left
    (1, 0, 0),      # move right
    (0, 0, 1),      # shoot
]
A_1_NONE, A_1_LEFT, A_1_RIGHT, A_1_FORWARD, A_1_SHOOT = 0, 1, 2, 3, 4
A_2_NONE, A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT, A_2_SHOOT = 0, 1, 2, 3, 4, 5
ALL_ACTIONS = [A_1_NONE, A_1_LEFT, A_1_RIGHT, A_1_FORWARD, A_1_SHOOT,
               A_2_NONE, A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT, A_2_SHOOT]


@dataclass
class StepResult:
    next_state: Tuple
    reward: float
    done: bool
    info: dict


class Arena:
    def __init__(self, arena_size: Tuple[int, int] = (800, 800), difficulty: int = 0):
        self.arena_size = arena_size
        self.difficulty = difficulty
        
        # Destruction objectives
        self.spawners: List[Tuple[float, float]] = []
        self.spawner_index: Dict[Tuple[float, float], int] = {}
        self.start: Tuple[float, float] = (0, 0)

        # State variables (initialized in reset)
        self.agent_pos: Tuple[float, float] = (0, 0)
        self.agent_angle: float = 0.0
        self.enemies: List[Tuple[float, float]] = []
        
        self.spawner_mask: int = 0
        self.alive: bool = True
        self.step_count: int = 0