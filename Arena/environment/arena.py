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
    """A Player specialized for training"""
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
NO_TARGET_POS = float('inf')


class Arena:
    def __init__(self, arena_size: Tuple[int, int] = (ARENA_WIDTH, ARENA_HEIGHT), difficulty: int = 0):
        # Arena info
        self.arena_size = arena_size
        self.difficulty = difficulty
        self.start: Tuple[float, float] = (0, 0)
        # These lists self-update when a new instance is created
        self.hittables: List[entities.Hittable] = []
        self.bullets: List[entities.Bullet] = []

        # Physics helper
        self.last_physic_frame = pygame.time.get_ticks()

        # State variables (initialized in reset)
        self.agent: Agent = Agent(self.start, angle=0.0)
        self.alive: bool = True
        self.step_count: int = 0

        # Spawn indication
        self.out_of_spawners = -999
        self.teleporters: List[entities.Fabricator] = [entities.Teleporter(pos, 0, env=self) for pos in self.select_spawners_positions()]
        # Destruction objectives
        self.try_spawning_spawners()
        self.spawners: List[entities.Spawner] = [spn for spn in self.hittables if isinstance(spn, entities.Spawner)]
        self.enemies: List[entities.Enemy] = [enem for enem in self.hittables if isinstance(enem, entities.Enemy)]

    def reset(self) -> Tuple:
        self.hittables.clear()
        self.bullets.clear()

        self.last_physic_frame = pygame.time.get_ticks()

        self.agent = Agent(self.start, angle=0.0)
        self.alive = True
        self.step_count = 0

        self.out_of_spawners = -999
        self.teleporters = [entities.Teleporter(pos, 0, env=self) for pos in self.select_spawners_positions()]
        self.try_spawning_spawners()
        self.spawners = [spn for spn in self.hittables if isinstance(spn, entities.Spawner)]
        self.enemies = [enem for enem in self.hittables if isinstance(enem, entities.Enemy)]

        return self.encode_state()

    def select_spawners_positions(self) -> List[Tuple[float, float]]:
        """Randomly select positions to spawn spawners"""
        amount = self.difficulty + 1
        spawn_padding = 80
        positions: List[Tuple] = []
        for i in range(amount):
            rand_pos = None
            # Select a random position at least 160 units away from agent position
            while vectorHelper.vec_len(rand_pos, self.agent.position) < spawn_padding * 2:
                rand_x = random.randint(80, ARENA_WIDTH - spawn_padding)
                rand_y = random.randint(80, ARENA_HEIGHT - spawn_padding)
                rand_pos = (float(rand_x), float(rand_y))
            if rand_pos is not None:
                positions.append(rand_pos)
        return positions
    
    def try_spawning_spawners(self):
        """Try spawning spawner with teleporters"""
        if len(self.teleporters) == 0:
            return
        for tper in self.teleporters:
            tper.try_spawn_with_cooldown(tper.pos, entities.SPAWN_SPAWNER, self.difficulty, self.agent)
        self.difficulty += 1

    def encode_state(self) -> Tuple:
        """
        Returns a tuple representing the current state of the Arena\n
        - Player position (tuple, seperate x, y)\n
        - Player velocity (tuple, seperate x, y)\n
        - Player orientation (tuple, seperate x, y)\n
            - orientation is shot direction\n
        - Distance and direction to nearest enemy (float and tuple, seperate x, y)\n
        - Distance and direction to nearest spawner (float and tuple, seperate x, y)\n
        - Distance and direction to nearest enemy bullet (float and tuple, seperate x, y)\n
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


        enemy_bullets = [bullet for bullet in self.bullets if bullet.owner != self.agent]
        closest_enemy_bullet: entities.Spawner = None
        closest_enemy_bullet_dist: float = float('inf')
        for bullet in enemy_bullets:
            dist = vectorHelper.vec_len(self.agent.position, bullet.position)
            if dist < closest_enemy_bullet_dist:
                closest_enemy_bullet_dist = dist
                closest_enemy_bullet = bullet

        
        state = (self.agent.position[0], self.agent.position[1], self.agent.velocity[0], self.agent.velocity[1],
                 agent_pointing[0], agent_pointing[1],
                 closest_enemy_dist,
                 closest_enemy.position[0] if closest_enemy is not None else NO_TARGET_POS,
                 closest_enemy.position[1] if closest_enemy is not None else NO_TARGET_POS,
                 closest_spawner_dist,
                 closest_spawner.position[0] if closest_spawner is not None else NO_TARGET_POS,
                 closest_spawner.position[1] if closest_spawner is not None else NO_TARGET_POS,
                 closest_enemy_bullet_dist,
                 closest_enemy_bullet.position[0] if closest_spawner is not None else NO_TARGET_POS,
                 closest_enemy_bullet.position[1] if closest_spawner is not None else NO_TARGET_POS,
                 self.agent.health, self.agent.max_health, self.agent.power, self.difficulty)
        return state

    def update(self):
        # Ensure physics work correctly using dt (in case of lag/delay)
        current_time = pygame.time.get_ticks()
        dt = (current_time - self.last_physic_frame) / 1000.0
        for b in self.bullets[:]:
            b.update(dt)
        for h in self.hittables[:]:
            h.update(dt)
        
        # Updating frame
        self.spawners: List[entities.Spawner] = [spn for spn in self.hittables if isinstance(spn, entities.Spawner)]
        self.enemies: List[entities.Enemy] = [enem for enem in self.hittables if isinstance(enem, entities.Enemy)]
        self.alive = not self.agent.out_of_health()

        # If there are no more spawners
        if len(self.spawners) == 0:
            # Only happens if teleporters list is empty
            if len(self.teleporters) == 0:
                # Time that spawners ran out
                self.out_of_spawners = pygame.time.get_ticks()
                # Populate teleporters list
                self.teleporters = [entities.Teleporter(pos, 500, current_time, self) for pos in self.select_spawners_positions()]
            # Try spawning spawner
            self.try_spawning_spawners()
        else:
            # Clear teleporter list if there are still spawners
            self.teleporters.clear()

    
    def step(self, style = SPEEN_AND_VROOM, action = A_NONE) -> StepResult:
        """
        TODO: Reward function
        """
        # Every single step is a physic frame, meaning the Agent will perform an action every frame

        self.agent.do(style=style, action=action)   # Perform an action
        
        self.update()

        # TODO: Reward function

        # Placeholder return
        return StepResult(
            next_state=self.encode_state(),
            reward=0.0,
            done=False,
            info={}
        )