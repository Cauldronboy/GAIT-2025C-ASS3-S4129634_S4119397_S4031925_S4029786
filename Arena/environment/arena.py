"""
Core Arena environment for reinforcement learning.
Polygonkind is dead
Area is fuel
GPU is full
"""

import pygame
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, Optional
import vectorHelper
import entities


# Action sets
SPEEN_AND_VROOM, BORING_4D_PAD = 0, 1
A_NONE, A_SHOOT = 0, 10
ANTI_CLOCKWISE, CLOCKWISE = 1, -1
A_1_FORWARD, A_1_LEFT, A_1_RIGHT = 1, 2, 3
A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT = 1, 2, 3, 4

class Agent(entities.Player):
    def do(self, style, action):
        """Perform an action"""
        if action == A_NONE: return
        if action == A_SHOOT: self.shoot()
        if style == SPEEN_AND_VROOM:
            if action == A_1_FORWARD:
                self.activate_thrust()
            elif action == A_1_LEFT:
                self.rotate(ANTI_CLOCKWISE)
            elif action == A_1_RIGHT:
                self.rotate(CLOCKWISE)
        elif style == BORING_4D_PAD:
            if action == A_2_UP:
                self.inertial_manipulator_up()
            elif action == A_2_DOWN:
                self.inertial_manipulator_down()
            elif action == A_2_LEFT:
                self.inertial_manipulator_left()
            elif action == A_2_RIGHT:
                self.inertial_manipulator_right()

@dataclass
class StepResult:
    next_state: Tuple
    reward: float
    done: bool
    info: dict


# Interaction variables
ARENA_WIDTH = 1000
ARENA_HEIGHT = 1000
ARENA_CORNERS = {
    "topleft": (0, 0),
    "topright": (1000, 0),
    "bottomright": (1000, 1000),
    "bottomleft": (0, 1000)
}
hittables: List[entities.Hittable] = []
bullets: List[entities.Bullet] = []


class Arena:
    def __init__(self, arena_size: Tuple[int, int] = (ARENA_WIDTH, ARENA_HEIGHT), difficulty: int = 0):
        self.arena_size = arena_size
        self.difficulty = difficulty
        
        # Destruction objectives
        self.spawners: List[entities.Spawner] = [spn for spn in hittables if isinstance(spn, entities.Spawner)]
        self.enemies: List[entities.Enemy] = [enem for enem in hittables if isinstance(enem, entities.Enemy)]

        self.start: Tuple[float, float] = (0, 0)

        # Physics helper
        self.last_physic_frame = pygame.time.get_ticks()

        # State variables (initialized in reset)
        self.agent: Agent = Agent(self.start, angle=0.0)
        self.alive: bool = True
        self.step_count: int = 0

        self.spawn_spawners()

    def reset(self) -> Tuple:
        hittables.clear()
        bullets.clear()
        self.agent = Agent(self.start, angle=0.0)
        self.spawner_mask = 0
        self.alive = True
        self.spawn_spawners()

        return self.encode_state()

    def spawn_spawners(self):
        amount = self.difficulty + 1
        spawn_padding = 80
        for i in range(amount):
            rand_x = random.randint(80, ARENA_WIDTH - spawn_padding)
            rand_y = random.randint(80, ARENA_HEIGHT - spawn_padding)
            rand_pos = (rand_x, rand_y)
            entities.Spawner(rand_pos, self.difficulty)

    def encode_state(self) -> Tuple:
        """
        Returns a tuple representing the current state of the Arena\n
        - Player position (tuple, seperate x, y)\n
        - Player velocity (tuple, seperate x, y)\n
        - Player orientation (tuple, seperate x, y)\n
            - orientation is shot direction\n
        - Distance and direction to nearest enemy (float and tuple, seperate x, y)\n
        - Distance and direction to nearest spawner (float and tuple, seperate x, y)\n
        - Player health (int)\n
        - Player max health (int)\n
        - Player power (int)\n
        - Current difficulty (int, start at 0)
        """
        agent_pointing = vectorHelper.ang_to_vec(self.agent.angle)

        closest_enemy: entities.Enemy = None
        closest_enemy_dist: float = float('inf')
        for enem in self.enemies:
            dist = vectorHelper.vec_len(self.agent.position, enem.position)
            if dist < closest_enemy_dist:
                closest_enemy_dist = dist
                closest_enemy = enem
        
        
        closest_spawner: entities.Spawner = None
        closest_spawner_dist: float = float('inf')
        for spn in self.spawners:
            dist = vectorHelper.vec_len(self.agent.position, spn.position)
            if dist < closest_spawner_dist:
                closest_spawner_dist = dist
                closest_spawner = spn

        
        state = (self.agent.position[0], self.agent.position[1], self.agent.velocity[0], self.agent.velocity[1],
                 agent_pointing[0], agent_pointing[1],
                 closest_enemy_dist,
                 closest_enemy.position[0] if closest_enemy is not None else ...,
                 closest_enemy.position[1] if closest_enemy is not None else ...,
                 closest_spawner_dist,
                 closest_spawner.position[0] if closest_spawner is not None else ...,
                 closest_spawner.position[1] if closest_spawner is not None else ...,
                 self.agent.health, self.agent.max_health, self.agent.power, self.difficulty)
        return state
    
    def step(self, style = SPEEN_AND_VROOM, action = A_NONE) -> StepResult:
        """
        TODO: Reward function
        """
        # Every single step is a physic frame, meaning the Agent will perform an action every frame

        self.agent.do(style=style, action=action)   # Perform an action

        # Ensure physics work correctly using dt (in case of lag/delay)
        current_physics_frame = pygame.time.get_ticks()
        dt = (current_physics_frame - self.last_physic_frame) / 1000.0
        for b in bullets[:]:
            b.update(dt)
        for h in hittables[:]:
            h.update(dt)
        
        self.spawners: List[entities.Spawner] = [spn for spn in hittables if isinstance(spn, entities.Spawner)]
        self.enemies: List[entities.Enemy] = [enem for enem in hittables if isinstance(enem, entities.Enemy)]

        self.alive = not self.agent.out_of_health()

        # Placeholder return
        return StepResult(
            next_state=self.encode_state(),
            reward=0.0,
            done=False,
            info={}
        )