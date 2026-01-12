"""Pygame renderer for Arena visualization."""

import pygame
import math
from typing import Optional, Tuple, List
import entities
import longinus
from .arena import Arena
import vectorHelper

RED     = (255, 0  , 0  )
GREEN   = (0  , 255, 0  )
BLUE    = (0  , 0  , 255)

def change_color_brightness(rgb: Tuple[int, int, int], per: int|float = 100) -> Tuple[int, int, int]:
    """Return an RGB tuple with brightness of `per`%"""
    max_in = max(rgb)
    max_rgb = (int(round(col * 255 / max_in)) for col in rgb)
    return tuple([int(round(col * per)) for col in max_rgb]) if max_in != 0 else (int(round(per * 255 / 100)), 0, 0)

def change_color_saturation(rgb: Tuple[int, int, int], per: int|float = 100) -> Tuple[int, int, int]:
    """Return an RGB tuple with saturation of `per`%"""
    max_in = max(rgb)
    min_in = min(rgb)
    c_max_in = max_in / 255 * 100
    c_min_in = min_in / 255 * 100
    d_col_in = c_max_in - c_min_in
    sat_in = d_col_in / c_max_in if c_max_in != 0 else 0
    sat_ratio = per / sat_in
    return tuple([int(round(max_in - (max_in - col) * sat_ratio)) for col in rgb]) if sat_in != 0 else (255, int(round(per * 255 / 100)), int(round(per * 255 / 100)))


class ArenaRenderer:
    """Renderer for Arena"""
    COL_BG          = (25 , 28 , 34 )
    COL_AGENT       = (74 , 222, 128)
    COL_ENEMIES     = {
        entities.EnemyTypes.RAMMER:                 (222, 74 , 74 ),
        entities.EnemyTypes.TANKIER_RAMMER:         (222, 111, 74 ),
        entities.EnemyTypes.EXPLOSIVE_RAMMER:       (222, 148, 74 ),
        entities.EnemyTypes.GOTTAGOFAST:            (165, 24 , 24 ),
        entities.EnemyTypes.PEW_PEW:                (147, 129, 129),
        entities.EnemyTypes.BIG_PEW_PEW:            (128, 112, 112),
        entities.EnemyTypes.SPAWNCEPTION:           (222, 74 , 148),
        entities.EnemyTypes.DIFFICULTY_LONGINUS:    (75 , 0  , 110)
    }
    COL_SPAWNER     = (222, 74 , 222)
    COL_TEXT        = (240, 240, 240)
    COL_TEXT_DIM    = (156, 163, 175)

    def __init__(self):
        self.screen = Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
    
    def init_display(self, env: Arena, title: str = "Never gonna give you up"):
        """Initialize the renderer"""
        pygame.init()
        self.screen = pygame.display.set_mode(env.size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 14)

    def close(self):
        """Close the renderer"""
        if self.screen:
            pygame.quit()

    def draw_regular_polygon(self, position: Tuple[float, float], n: int = 3, angle: float = 0.0, radius: float = 2.0,
                             color: Tuple[int, int, int, Optional[int]] = (0, 230, 0), line_width: int = 0) -> List[Tuple[float, float]]:
        """
        Draw a regular polygon at `position` with one vertex pointing at `angle` (degrees)\n
        Each vertex is `radius` distance away from `position`
        `color`: (red, green, blue, opacity)
        """
        vertices: List[Tuple[float, float]] = []
        start_point = (radius, 0.0)
        start_point = vectorHelper.vec_rotate(start_point, angle, position)
        angle_between_vertices = 360.0 / n
        for i in range(n):
            vertex = vectorHelper.vec_rotate(start_point, angle_between_vertices * i, position)
            vertices.append(vertex)
        
        pygame.draw.polygon(self.screen, color, vertices, line_width)

    def huskify(self, col_variable):
        """Darkens the color"""
        return change_color_brightness(col_variable, 20)

    def draw_player(self, env: Arena):
        """Draw the player"""
        if env.alive:
            self.draw_regular_polygon(env.agent.position, 3, env.agent.angle, env.agent.hitbox.width*1.25, self.COL_AGENT)
        else:
            # Draw husk
            self.draw_regular_polygon(env.agent.position, 3, env.agent.angle, env.agent.hitbox.width*1.25,
                                      self.huskify(self.COL_AGENT))
            
    def draw_hittables(self, env: Arena):
        """Draw everything in the `hittables` list"""
        for htb in env.hittables:
            # Draw player
            if isinstance(htb, entities.Player):
                self.draw_player(env)
            # Draw spawners
            elif isinstance(htb, entities.Spawner):
                pygame.draw.circle(self.screen, self.COL_SPAWNER, htb.position,
                                   htb.hitbox.width/math.sqrt(2))
            # Draw enemies
            elif isinstance(htb, entities.Enemy):
                point_amount = htb.type.value + 3
                point_distance = htb.hitbox.width * 1.25 if point_amount == 3 else 0.75 if point_amount == 4 else 0.7
                self.draw_regular_polygon(htb.position, point_amount, htb.angle, point_distance,
                                          self.COL_ENEMIES[htb.type])
            # Draw husks
            elif isinstance(htb, entities.Husk):
                point_amount = htb.type.value + 3
                point_distance = htb.hitbox.width * 1.25 if point_amount == 3 else 0.75 if point_amount == 4 else 0.7
                self.draw_regular_polygon(htb.position, point_amount, htb.angle, point_distance,
                                          change_color_brightness(self.COL_ENEMIES[htb.type], 20))
                
    def draw_bullets(self, env: Arena):
        """Draw everything from the `bullets` list"""
        for b in env.bullets:
            # Draw normal bullets
            if not isinstance(b, longinus.Danmaku):
                # Outer edge
                pygame.draw.circle(self.screen, change_color_saturation(RED, 100), b.position, b.hitbox.width/2)
                # Mantle
                pygame.draw.circle(self.screen, change_color_saturation(RED, 200 / 3), b.position, b.hitbox.width * 3/8)
                # Outer core
                pygame.draw.circle(self.screen, change_color_saturation(RED, 100 / 3), b.position, b.hitbox.width/4)
                # Core
                pygame.draw.circle(self.screen, (255, 255, 255), b.position, b.hitbox.width/8)
            else:
                # Outer edge
                pygame.draw.circle(self.screen, change_color_saturation(b.color, 100), b.position, b.hitbox.width/2)
                # Mantle
                pygame.draw.circle(self.screen, change_color_saturation(b.color, 200 / 3), b.position, b.hitbox.width * 3/8)
                # Outer core
                pygame.draw.circle(self.screen, change_color_saturation(b.color, 100 / 3), b.position, b.hitbox.width/4)
                # Core
                pygame.draw.circle(self.screen, (255, 255, 255), b.position, b.hitbox.width/8)
    
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
            extra_info if extra_info else "R: reset | ESC: quit"
        ]
        
        for i, text in enumerate(lines):
            if i == 0:
                # Title in brighter color
                surface = self.font.render(text, True, self.COL_TEXT)
            else:
                surface = self.font_small.render(text, True, self.COL_TEXT_DIM)
            self.screen.blit(surface, (10, y_offset + i * line_height))