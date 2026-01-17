"""
Core Arena environment for reinforcement learning.\n
f`Polygonkind is dead\nArea is fuel\nGPU is full`
"""

import pygame
import math
import random
import enum
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, Optional

try:
    from . import vectorHelper
    from .renderer import ArenaRenderer
except ImportError:
    import vectorHelper
    from renderer import ArenaRenderer

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Action sets
SPEEN_AND_VROOM, BORING_4D_PAD = 0, 1
A_NONE, A_SHOOT = 0, 10
ANTI_CLOCKWISE, CLOCKWISE = -1, 1
A_1_FORWARD, A_1_LEFT, A_1_RIGHT = 1, 2, 3
A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT = 1, 2, 3, 4
SPEEN_VROOM_ALL_ACTIONS = [A_NONE, A_1_FORWARD, A_1_LEFT, A_1_RIGHT, A_SHOOT]
BORING_4D_PAD_ALL_ACTIONS = [A_NONE, A_2_UP, A_2_DOWN, A_2_LEFT, A_2_RIGHT, A_SHOOT]
ALL_ACTIONS = [SPEEN_VROOM_ALL_ACTIONS, BORING_4D_PAD_ALL_ACTIONS]


# Interaction variables
ARENA_WIDTH = 800
ARENA_HEIGHT = 800
ARENA_CORNERS = {
    "topleft": (0, 0),
    "topright": (1000, 0),
    "bottomright": (1000, 1000),
    "bottomleft": (0, 1000)
}
NO_TARGET_POS = 10000.0  # Large finite value indicating no target found


class ArenaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, control_style=SPEEN_AND_VROOM, size: Tuple[int, int] = (ARENA_WIDTH, ARENA_HEIGHT), difficulty: int = 0, render_mode: str = None):
        # Arena info
        self.size = size
        self.difficulty = difficulty
        self.start: Tuple[float, float] = (ARENA_WIDTH / 2, ARENA_HEIGHT / 2)
        self.render_mode = render_mode
        self.control_style = control_style

        self.max_steps = 1000
        
        # Define action and observation spaces
        # Actions: 0=no action, 1=shoot, 2=forward, 3=left, 4=right, 5=backward
        if self.control_style == SPEEN_AND_VROOM:
            self.action_space = spaces.Discrete(5)  # 0-4 (no unused action)
        else:
            self.action_space = spaces.Discrete(6)  # 0-5
        
        # Observation space: 19 elements
        # [pos_x, pos_y, vel_x, vel_y, point_x, point_y, 
        #  enemy_dist, enemy_pos_x, enemy_pos_y,
        #  spawner_dist, spawner_pos_x, spawner_pos_y,
        #  bullet_dist, bullet_pos_x, bullet_pos_y,
        #  health, max_health, power, difficulty]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 19, dtype=np.float32),
            high=np.array([np.inf] * 19, dtype=np.float32),
            dtype=np.float32
        )
        
        # These lists self-update when a new instance is created
        self.hittables: List = []
        self.bullets: List = []

        # Physics helper
        self.last_physic_frame = pygame.time.get_ticks()

        # State variables (initialized in reset)
        self.agent = None
        self.score = 0
        self.alive: bool = True
        self.step_count: int = 0

        # Spawn indication
        self.out_of_spawners = -999
        self.teleporters: List = []
        self.spawners: List = []
        self.enemies: List = []
        
        # Renderer setup
        self.renderer: Optional[ArenaRenderer] = None
        if self.render_mode == "human":
            self.renderer = ArenaRenderer()
            self.renderer.init_display(self)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.hittables.clear()
        self.bullets.clear()

        self.last_physic_frame = pygame.time.get_ticks()

        # Initialize agent (import here to avoid circular imports)
        try:
            from . import entities
        except ImportError:
            import entities
        
        self.agent = entities.Agent(self.start, angle=0.0, env=self)
        self.score = 0
        self.alive = True
        self.step_count = 0

        self.out_of_spawners = -999
        self.teleporters = [entities.Teleporter(pos, 0, env=self) for pos in self.select_spawners_positions()]
        self.try_spawning_spawners()
        self.spawners = [spn for spn in self.hittables if isinstance(spn, entities.Spawner)]
        self.enemies = [enem for enem in self.hittables if isinstance(enem, entities.Enemy)]

        observation = self._get_observation()
        return observation, {}

    def select_spawners_positions(self) -> List[Tuple[float, float]]:
        """Randomly select positions to spawn spawners"""
        amount = self.difficulty + 1
        spawn_padding = 80
        positions: List[Tuple] = []
        for i in range(amount):
            rand_x = random.randint(80, ARENA_WIDTH - spawn_padding)
            rand_y = random.randint(80, ARENA_HEIGHT - spawn_padding)
            rand_pos = (float(rand_x), float(rand_y))
            # Select a random position at least 160 units away from agent position
            while vectorHelper.vec_len(rand_pos, self.agent.position) < spawn_padding * 2:
                rand_x = random.randint(80, ARENA_WIDTH - spawn_padding)
                rand_y = random.randint(80, ARENA_HEIGHT - spawn_padding)
                rand_pos = (float(rand_x), float(rand_y))
            if rand_pos is not None:
                positions.append(rand_pos)
        return positions
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as numpy array"""
        state = self.encode_state()
        return np.array(state, dtype=np.float32)
    
    def try_spawning_spawners(self):
        """Try spawning spawner with teleporters"""
        if len(self.teleporters) == 0:
            return
        for tper in self.teleporters:
            tper.try_spawn_with_cooldown(tper.pos, entities.SPAWN_SPAWNER, self.difficulty, self.agent)

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
        closest_enemy_dist: float = 10000.0
        for enem in self.enemies:
            dist = vectorHelper.vec_len(self.agent.position, enem.position)
            if dist < closest_enemy_dist:
                closest_enemy_dist = dist
                closest_enemy = enem
        
        
        closest_spawner: entities.Spawner = None
        closest_spawner_dist: float = 10000.0
        for spn in self.spawners:
            dist = vectorHelper.vec_len(self.agent.position, spn.position)
            if dist < closest_spawner_dist:
                closest_spawner_dist = dist
                closest_spawner = spn


        enemy_bullets = [bullet for bullet in self.bullets if bullet.owner != self.agent]
        closest_enemy_bullet: entities.Spawner = None
        closest_enemy_bullet_dist: float = 10000.0
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
                 closest_enemy_bullet.position[0] if closest_enemy_bullet is not None else NO_TARGET_POS,
                 closest_enemy_bullet.position[1] if closest_enemy_bullet is not None else NO_TARGET_POS,
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

        # Skip the rest if not alive
        if not self.alive:
            return
        # If there are no more spawners
        if len(self.spawners) == 0:
            # Only happens if teleporters list is empty
            if len(self.teleporters) == 0:
                self.difficulty += 1
                # Time that spawners ran out
                self.out_of_spawners = pygame.time.get_ticks()
                # Populate teleporters list
                self.teleporters = [entities.Teleporter(pos, 1000, current_time, self) for pos in self.select_spawners_positions()]
            # Try spawning spawner
            self.try_spawning_spawners()
        else:
            # Clear teleporter list if there are still spawners
            self.teleporters.clear()
        self.last_physic_frame = current_time
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Integer ranging from 0 to\n
                control_style = SPEEN_AND_VROOM: 4\n
                control_style = BORING_4D_PAD: 5
        
        Returns:
            observation: np.ndarray of current state
            reward: float reward for this step
            terminated: bool whether episode ended (agent dead)
            truncated: bool whether episode was truncated (time limit)
            info: dict with additional info
        """
        
        previous_score = self.score
        previous_hp = self.agent.health
        previous_maxhp = self.agent.max_health
        previous_difficulty = self.difficulty

        # Perform action
        self.agent.do(style=self.control_style, action=ALL_ACTIONS[self.control_style][action])
        
        # Update environment
        self.update()

        reward = 0.0
        done = False
        cd = False


        # TODO: Reward function

        if self.agent.health <= self.agent.max_health: # Agent loses 1/4 reward every second to discourage running
            reward -= 0.001
        
        if previous_hp > self.agent.health: # Agent loses 1 reward if hit
            reward -= 1
        
        if previous_maxhp < self.agent.max_health: # Incentivize overheal
            reward += (self.agent.max_health - previous_maxhp) * 5
        
        if self.agent.out_of_health() == True: # Don't die
            reward -= 20
            
        if previous_difficulty < self.difficulty: # Reward for moving to next stage
            reward += 10 

        # Add reward according to score increase
        score_diff = self.score - previous_score
        if score_diff > 0:
            reward += (score_diff / 10)
            score_diff = 0
        
        
        # Get new observation
        observation = self._get_observation()

        truncated = self.step_count >= self.max_steps
        terminated = not self.alive
            
        if terminated:
            print("Episode ended: TERMINATED")
        if truncated:
            print("Episode ended: TRUNCATED")
        
        info = {"step_count": self.step_count}
        
        if not terminated:
            # Add reward for each enemies hit within the last 100 ms
            for enem in self.enemies[:]:
                if enem.invincible:
                    reward += 1.0

        self.step_count += 1
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = ArenaRenderer()
                self.renderer.init_display(self)
            self.renderer.render(self, step=self.step_count)
            self.renderer.tick(self.metadata["render_fps"])
    
    def close(self):
        """Close the renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Import entities after ArenaEnv class is defined to avoid circular imports
try:
    from . import entities
except ImportError:
    import entities
 
