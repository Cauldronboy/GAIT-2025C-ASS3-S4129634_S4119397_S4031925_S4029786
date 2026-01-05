"""Pygame renderer for GridWorld visualization."""

import pygame
from typing import Tuple, Optional
from .gridworld import GridWorld


class GridWorldRenderer:
    COL_BG = (25, 28, 34)
    COL_GRID = (45, 50, 58)
    COL_AGENT = (74, 222, 128)
    COL_APPLE = (252, 92, 101)
    COL_KEY = (250, 204, 21)
    COL_CHEST_CLOSED = (168, 85, 247)
    COL_CHEST_OPEN = (100, 50, 150)
    COL_ROCK = (120, 113, 108)
    COL_FIRE = (251, 146, 60)
    COL_MONSTER = (239, 68, 68)
    COL_TEXT = (240, 240, 240)
    COL_TEXT_DIM = (156, 163, 175)
    
    def __init__(self, tile_size: int = 48):
        self.tile_size = tile_size
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        
    def init_display(self, env: GridWorld, title: str = "GridWorld RL"):
        pygame.init()
        width = env.w * self.tile_size
        height = env.h * self.tile_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 14)
    
    def close(self):
        if self.screen:
            pygame.quit()
    
    def draw_grid(self, env: GridWorld):
        for x in range(env.w):
            for y in range(env.h):
                rect = pygame.Rect(
                    x * self.tile_size,
                    y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.screen, self.COL_GRID, rect, 1)
    
    def draw_tile_centered_circle(self, pos: Tuple[int, int], color: Tuple[int, int, int], 
                                   radius_ratio: float = 0.3):
        x, y = pos
        center = (
            x * self.tile_size + self.tile_size // 2,
            y * self.tile_size + self.tile_size // 2
        )
        radius = int(self.tile_size * radius_ratio)
        pygame.draw.circle(self.screen, color, center, radius)
    
    def draw_tile_rect(self, pos: Tuple[int, int], color: Tuple[int, int, int], 
                       margin: int = 4):
        """Draw a rectangle filling a tile with margin."""
        x, y = pos
        rect = pygame.Rect(
            x * self.tile_size + margin,
            y * self.tile_size + margin,
            self.tile_size - 2 * margin,
            self.tile_size - 2 * margin
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
    
    def draw_rocks(self, env: GridWorld):
        """Draw rock obstacles."""
        for rock in env.rocks:
            self.draw_tile_rect(rock, self.COL_ROCK, margin=2)
    
    def draw_fires(self, env: GridWorld):
        for fire in env.fires:
            self.draw_tile_centered_circle(fire, self.COL_FIRE, 0.25)
            x, y = fire
            cx = x * self.tile_size + self.tile_size // 2
            cy = y * self.tile_size + self.tile_size // 2
            
            pygame.draw.circle(self.screen, (251, 191, 36), (cx, cy), 
                             int(self.tile_size * 0.35), 2)
    
    def draw_apples(self, env: GridWorld):
        for apple_pos, idx in env.apple_index.items():
            if (env.apple_mask >> idx) & 1:
                self.draw_tile_centered_circle(apple_pos, self.COL_APPLE, 0.28)
    
    def draw_keys(self, env: GridWorld):
        for key in env.keys:
            x, y = key
            cx = x * self.tile_size + self.tile_size // 2
            cy = y * self.tile_size + self.tile_size // 2
            
            rect = pygame.Rect(cx - 3, cy - 8, 6, 12)
            pygame.draw.rect(self.screen, self.COL_KEY, rect, border_radius=2)
            pygame.draw.circle(self.screen, self.COL_KEY, (cx, cy - 10), 6)
    
    def draw_chests(self, env: GridWorld):
        for chest_pos, idx in env.chest_index.items():
            if (env.chest_mask >> idx) & 1:
                self.draw_tile_rect(chest_pos, self.COL_CHEST_CLOSED, margin=8)
                x, y = chest_pos
                cx = x * self.tile_size + self.tile_size // 2
                cy = y * self.tile_size + self.tile_size // 2
                pygame.draw.circle(self.screen, self.COL_KEY, (cx, cy), 4)
            else:
                # Opened chest (dimmed)
                self.draw_tile_rect(chest_pos, self.COL_CHEST_OPEN, margin=8)
    
    def draw_monsters(self, env: GridWorld):
        for monster in env.monsters:
            self.draw_tile_centered_circle(monster, self.COL_MONSTER, 0.32)
            x, y = monster
            cx = x * self.tile_size + self.tile_size // 2
            cy = y * self.tile_size + self.tile_size // 2
            eye_offset = 5
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             (cx - eye_offset, cy - 3), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             (cx + eye_offset, cy - 3), 3)
    
    def draw_agent(self, env: GridWorld):
        if env.alive:
            self.draw_tile_rect(env.agent, self.COL_AGENT, margin=8)
        else:
            x, y = env.agent
            cx = x * self.tile_size + self.tile_size // 2
            cy = y * self.tile_size + self.tile_size // 2
            offset = self.tile_size // 4
            pygame.draw.line(self.screen, (150, 150, 150), 
                           (cx - offset, cy - offset), 
                           (cx + offset, cy + offset), 3)
            pygame.draw.line(self.screen, (150, 150, 150), 
                           (cx + offset, cy - offset), 
                           (cx - offset, cy + offset), 3)
    
    def draw_hud(self, episode: int, total_episodes: int, step: int, 
                 epsilon: float, episode_reward: float, algorithm: str = "Q-Learning",
                 level_name: str = "Level 0", extra_info: str = ""):
        """Draw heads-up display with training stats."""
        y_offset = 8
        line_height = 20
        
        lines = [
            f"{algorithm} - {level_name}",
            f"Episode: {episode + 1}/{total_episodes}  Step: {step}",
            f"Epsilon: {epsilon:.3f}  Reward: {episode_reward:.2f}",
            extra_info if extra_info else "V: fast mode | R: reset | ESC: quit"
        ]
        
        for i, text in enumerate(lines):
            if i == 0:
                # Title in brighter color
                surface = self.font.render(text, True, self.COL_TEXT)
            else:
                surface = self.font_small.render(text, True, self.COL_TEXT_DIM)
            self.screen.blit(surface, (10, y_offset + i * line_height))
    
    def render(self, env: GridWorld, episode: int = 0, total_episodes: int = 1000,
               step: int = 0, epsilon: float = 0.0, episode_reward: float = 0.0,
               algorithm: str = "Q-Learning", level_name: str = "Level 0",
               extra_info: str = ""):
        """
        Render complete frame.
        
        Args:
            env: GridWorld environment
            episode: Current episode number
            total_episodes: Total episodes to train
            step: Current step in episode
            epsilon: Current epsilon value
            episode_reward: Cumulative reward this episode
            algorithm: Algorithm name (Q-Learning or SARSA)
            level_name: Name of current level
            extra_info: Additional info to display
        """
        # Clear screen
        self.screen.fill(self.COL_BG)
        
        # Draw in layers (back to front)
        self.draw_grid(env)
        self.draw_rocks(env)
        self.draw_fires(env)
        self.draw_apples(env)
        self.draw_keys(env)
        self.draw_chests(env)
        self.draw_monsters(env)
        self.draw_agent(env)
        self.draw_hud(episode, total_episodes, step, epsilon, episode_reward,
                     algorithm, level_name, extra_info)
        
        # Update display
        pygame.display.flip()
    
    def tick(self, fps: int):
        """Control frame rate."""
        if self.clock:
            self.clock.tick(fps)
