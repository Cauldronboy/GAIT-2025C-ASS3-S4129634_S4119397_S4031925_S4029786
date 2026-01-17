"""Pygame renderer for Arena visualization."""

import pygame
import math
from typing import Optional, Tuple, List
import environment.entities as entities
import environment.longinus as longinus
from environment.arena import ArenaEnv as Arena
import environment.vectorHelper as vectorHelper

RED     = (255, 0  , 0  )
GREEN   = (0  , 255, 0  )
BLUE    = (0  , 0  , 255)
WHITE   = (255, 255, 255)
BLACK   = (0  , 0  , 0  )

def get_current_time(env: Arena):
    return env.step_count * 1000 / 60 if env is not None else 0

def change_color_brightness(rgb: Tuple[int, int, int], per: int|float = 100) -> Tuple[int, int, int]:
    """Return an RGB tuple with brightness of `per`%"""
    max_in = max(rgb)
    max_rgb = (int(round(col * 255 / max_in)) for col in rgb)
    return tuple([int(round(col * per / 100)) for col in max_rgb]) if max_in != 0 else (int(round(per * 255 / 100)), 0, 0)

def change_color_saturation(rgb: Tuple[int, int, int], per: int|float = 100) -> Tuple[int, int, int]:
    """Return an RGB tuple with saturation of `per`%"""
    max_in = max(rgb)
    min_in = min(rgb)
    c_max_in = max_in / 255
    c_min_in = min_in / 255
    d_col_in = c_max_in - c_min_in
    sat_in = d_col_in / c_max_in * 100 if c_max_in != 0 else 0
    sat_ratio = per / sat_in
    test = tuple([int(round(max_in - (max_in - col) * sat_ratio)) for col in rgb]) if sat_in != 0 else (255, int(round(per * 255 / 100)), int(round(per * 255 / 100)))
    return test


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
        self.screen: Optional[pygame.Surface] = None
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
        start_point = (radius + position[0], position[1])
        start_point = vectorHelper.vec_rotate(start_point, angle, position)
        angle_between_vertices = 360.0 / n
        for i in range(n):
            vertex = vectorHelper.vec_rotate(start_point, angle_between_vertices * i, position)
            vertices.append(vertex)
        
        pygame.draw.polygon(self.screen, color, vertices, line_width)

    def draw_health_bar(self, hittable: entities.Hittable):
        """Draw the health bar of hittable"""
        max_hp = hittable.max_health
        hp = hittable.health
        pos = hittable.position
        hp_bar_size = hittable.hitbox.width * 2
        hp_bar_left_top = (pos[0] - hp_bar_size / 2, pos[1] + hp_bar_size / 2 + 5)
        hp_display_size = hp_bar_size * (hp / max_hp if max_hp != 0 else 0)
        hp_bar = pygame.Rect(hp_bar_left_top, (hp_bar_size, 6))
        hp_display = pygame.Rect(hp_bar_left_top, (hp_display_size, 6))
        pygame.draw.rect(self.screen, change_color_brightness(BLACK, 25), hp_bar, border_radius=3)
        pygame.draw.rect(self.screen, GREEN, hp_display, border_radius=3)
        pygame.draw.rect(self.screen, BLACK, hp_bar, width=1, border_radius=3)

    def huskify(self, col_variable):
        """Darkens the color"""
        return change_color_brightness(col_variable, 20)

    def draw_player(self, env: Arena):
        """Draw the player"""
        if env.alive:
            self.draw_regular_polygon(env.agent.position, 3, env.agent.angle, env.agent.hitbox.width*1.25, WHITE if env.agent.invincible else self.COL_AGENT)
            self.draw_health_bar(env.agent)
        else:
            # Draw husk
            self.draw_regular_polygon(env.agent.position, 3, env.agent.angle, env.agent.hitbox.width*1.25,
                                      self.huskify(self.COL_AGENT))
            
    def draw_hittables(self, env: Arena):
        """Draw everything in the `hittables` list"""
        for htb in env.hittables:
            # Draw player
            if isinstance(htb, entities.Agent):
                self.draw_player(env)
            # Draw spawners
            elif isinstance(htb, entities.Spawner):
                self.draw_health_bar(htb)
                pygame.draw.circle(self.screen, WHITE if htb.invincible else self.COL_SPAWNER, htb.position,
                                   htb.hitbox.width/math.sqrt(2))
                point_amount = htb.spawn_type.value + 3
                point_distance = htb.hitbox.width / 2 * 1.25 if point_amount == 3 else htb.hitbox.width / 2 * 0.75 if point_amount == 4 else htb.hitbox.width / 2 * 0.7
                angle = get_current_time(env) / 50 + math.radians(env.hittables.index(htb))
                self.draw_regular_polygon(htb.position, point_amount, angle, point_distance,
                                          WHITE if htb.invincible else self.COL_ENEMIES[htb.spawn_type])
            # Draw enemies
            elif isinstance(htb, entities.Enemy):
                self.draw_health_bar(htb)
                point_amount = htb.type.value + 3
                point_distance = htb.hitbox.width * 1.25 if point_amount == 3 else htb.hitbox.width * 0.75 if point_amount == 4 else htb.hitbox.width * 0.7
                self.draw_regular_polygon(htb.position, point_amount, htb.angle, point_distance,
                                          WHITE if htb.invincible else self.COL_ENEMIES[htb.type])
            # Draw husks
            elif isinstance(htb, entities.Husk):
                if htb.type is not None:
                    point_amount = htb.type.value + 3
                    point_distance = htb.size * 1.25 if point_amount == 3 else htb.size * 0.75 if point_amount == 4 else htb.size * 0.7
                    self.draw_regular_polygon(htb.position, point_amount, htb.angle, point_distance,
                                            self.huskify(self.COL_ENEMIES[htb.type] if htb.type in self.COL_ENEMIES.keys() else self.COL_SPAWNER))
                else:
                    pygame.draw.circle(self.screen, self.huskify(self.COL_SPAWNER), htb.position,
                                   htb.size/math.sqrt(2))
                
    def draw_bullets(self, env: Arena):
        """Draw everything from the `bullets` list"""
        for b in env.bullets:
            # Draw normal bullets
            color = RED
            if isinstance(b, longinus.Danmaku):
                color = b.color
            # Outer edge
            pygame.draw.circle(self.screen, change_color_saturation(color, 100), b.position, b.hitbox.width/2)
            # Mantle
            pygame.draw.circle(self.screen, change_color_saturation(color, 200 / 3), b.position, b.hitbox.width * 3/8)
            # Outer core
            pygame.draw.circle(self.screen, change_color_saturation(color, 100 / 3), b.position, b.hitbox.width/4)
            # Core
            pygame.draw.circle(self.screen, WHITE, b.position, b.hitbox.width/8)
    
    def draw_teleporter(self, env: Arena):
        """Draw spawner spawn indicator"""
        for t in env.teleporters:
            # Skip if spawn_cooldown is 0 to avoid division by zero
            if t.spawn_cooldown == 0:
                continue
            
            passed = get_current_time(env) - t.started
            og_size = 40 * ((t.spawn_cooldown - passed / 2) / t.spawn_cooldown + 0.5)
            ratio = math.sqrt(3)/2
            smaller_r = og_size * ratio * 2 / 3
            fx = (((passed - t.spawn_cooldown + 300) / 250) ** 2) * 100
            gx = fx - (passed)
            hx = gx if fx > (fx - gx) and passed < 500 else 0
            qx = fx - hx
            color_function = 0 if qx < 0 else 100 if qx > 100 else qx
            color_saturation = min(100, max(0, color_function))
            color = change_color_saturation(RED, color_saturation)
            pygame.draw.circle(self.screen, color, t.pos, og_size + 8, width=1)
            pygame.draw.circle(self.screen, color, t.pos, smaller_r * 1.5, width=1)
            pygame.draw.circle(self.screen, color, t.pos, smaller_r, width=1)
            self.draw_regular_polygon(t.pos, 6, 90.0 + passed * 0.1, og_size + 8, color, 1)
            self.draw_regular_polygon(t.pos, 6, 90.0 - passed * 0.1, smaller_r * 1.5, color, 1)
            self.draw_regular_polygon(t.pos, 3, 90.0 + passed * 0.1, smaller_r, color, 1)
            self.draw_regular_polygon(t.pos, 3, 270.0 + passed * 0.1, smaller_r, color, 1)

    def draw_hud(self, episode: int, total_episodes: int, step: int,
                 algorithm: str = "Deep Reinforcement Learning",
                 difficulty: int = 0, agent = None, extra_info: str = ""):
        """Draw heads-up display with training stats."""
        y_offset = 8
        line_height = 20
        
        lines = [
            f"{algorithm} - Difficulty: {difficulty}",
            f"Episode: {episode + 1}/{total_episodes}  Step: {step}",
            f"Health: {agent.health}/{agent.max_health}  Damage: {agent.power}" if agent is not None else None,
            extra_info if extra_info else "R: reset | ESC: quit"
        ]
        
        for i, text in enumerate(lines):
            if i == 0:
                # Title in brighter color
                surface = self.font.render(text, True, self.COL_TEXT)
            else:
                surface = self.font_small.render(text, True, self.COL_TEXT_DIM)
            self.screen.blit(surface, (10, y_offset + i * line_height))

    def render(self, env: Arena, episode: int = 0, total_episodes: int = 1000,
               step: int = 0, algorithm: str = "Deep Reinforcement Learning",
               extra_info: str = ""):
        """
        Render complete frame
        
        :param env: Arena environment being simulated
        :type env: Arena
        :param episode: Current episode number
        :type episode: int
        :param total_episodes: Max episode number
        :type total_episodes: int
        :param step: Current step
        :type step: int
        :param algorithm: Algorithm being used
        :type algorithm: str
        :param extra_info: More info as needed
        :type extra_info: str
        """
         # Clear screen
        self.screen.fill(self.COL_BG)

        self.draw_teleporter(env)
        self.draw_hittables(env)
        self.draw_bullets(env)
        self.draw_hud(episode, total_episodes, step, algorithm, env.difficulty, env.agent, extra_info)

        # Update display
        pygame.display.flip()

    def tick(self, fps: int):
        """Control frame rate."""
        if self.clock:
            self.clock.tick(fps)