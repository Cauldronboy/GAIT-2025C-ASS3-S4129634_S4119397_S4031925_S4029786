"""Entities in the Arena environment."""

import pygame
import math
import random
import enum
from typing import Tuple, List, Set, Dict, Optional
import vectorHelper


# Object hitbox sizes
PLAYER_HITBOX_SIZE = 20
BULLET_HITBOX_SIZE = 4
ENEMY_HITBOX_SIZE = 20
SPAWNER_HITBOX_SIZE = 40


# Shape sweep collision detection
def rect_sweep(
    rect: pygame.Rect,
    vel: Tuple[float, float],
    obstacles: List["Hittable"],
    exceptions: List["Hittable"] = []
) -> Optional["Hittable"]:
    """
    Swept AABB collision detection.
    Assumes rect is at its starting position.
    Returns the first Hittable hit, or None.
    """

    vx, vy = vel
    earliest_t = 1.0
    hit_object = None

    not_friendlies = [obs for obs in obstacles if obs not in exceptions]

    for not_fren in not_friendlies:
        box = not_fren.hitbox

        # Select x entry and exit distances
        if vx > 0:
            x_entry = box.left - rect.right
            x_exit  = box.right - rect.left
        else:
            x_entry = box.right - rect.left
            x_exit  = box.left - rect.right

        # Select y entry and exit distances
        if vy > 0:
            y_entry = box.top - rect.bottom
            y_exit  = box.bottom - rect.top
        else:
            y_entry = box.bottom - rect.top
            y_exit  = box.top - rect.bottom

        # Calculate entry and exit times
        if vx == 0:
            tx_entry = -math.inf
            tx_exit  = math.inf
        else:
            tx_entry = x_entry / vx
            tx_exit  = x_exit / vx

        if vy == 0:
            ty_entry = -math.inf
            ty_exit  = math.inf
        else:
            ty_entry = y_entry / vy
            ty_exit  = y_exit / vy

        # Entry is when both axes have entered
        t_entry = max(tx_entry, ty_entry)
        # Exit is when either axis has exited
        t_exit  = min(tx_exit, ty_exit)

        # No collision
        if t_entry > t_exit or t_entry < 0 or t_entry > 1:
            continue

        if t_entry < earliest_t:
            earliest_t = t_entry
            hit_object = not_fren

    return hit_object


# Object classes
class Bullet:
    """Projectiles used by player and enemies"""
    def __init__(self, position: Tuple[float, float],
                 direction: Tuple[float, float],
                 owner: Optional["Hittable"] = None,
                 damage: int = 10,
                 speed: int = 20,
                 hittables: List["Hittable"] = []):
        self.owner = owner
        self.position = position
        self.hittables = hittables
        self.direction = vectorHelper.vec_norm(direction)
        self.speed = speed
        self.damage = damage
        self.hitbox = pygame.Rect(position[0] - BULLET_HITBOX_SIZE // 2,
                                  position[1] - BULLET_HITBOX_SIZE // 2,
                                  BULLET_HITBOX_SIZE,
                                  BULLET_HITBOX_SIZE)
    def hit(self, hittable: "Hittable"):
        hittable.take_damage(self.damage)
    def update(self, dt: float):
        movement = (self.direction[0] * self.speed * dt, self.direction[1] * self.speed * dt)
        gottem = rect_sweep(self.hitbox, movement, self.hittables, exceptions={self.owner} if self.owner else [])
        if gottem is not None:
            self.hit(gottem)
            return
        self.position = (self.position[0] + movement[0], self.position[1] + movement[1])
        self.hitbox.update(self.position[0] - 2, self.position[1] - 2, 4, 4)

class Explosion(Bullet):
    """Basically a stationary bullet that damages on contact"""
    def __init__(self, position,
                 owner = None,
                 damage = 50,
                 hittables = [],
                 radius = 20):
        super().__init__(position, (0, 0), owner, damage, 0, hittables)
        self.hitbox = pygame.Rect(position[0] - radius,
                                  position[1] - radius,
                                  radius * 2,
                                  radius * 2)
        self.life_expectancy = 500    # milliseconds
        self.start_time = pygame.time.get_ticks()
        self.radius = radius
    def update(self, dt: float):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time
        t = max(0.0, 1.0 - elapsed / self.life_expectancy)
        remaining_damage = self.damage * t
        if current_time - self.start_time >= self.life_expectancy:
            self.hitbox = None
            return
        for hittable in self.hittables:
            if hittable.hitbox and self.hitbox.colliderect(hittable.hitbox):

                hit_dir = vectorHelper.vec_sub(hittable.position, self.position)
                hit_dir = vectorHelper.vec_norm(hit_dir)
                
                dist = max(1e-6, vectorHelper.vec_len(self.position, hittable.position))
                falloff = min(1.0, ((self.radius / 2) / dist) ** 2)

                force = remaining_damage * falloff

                # knockback = impulse
                hittable.pushed((hit_dir[0] * force * dt,
                                hit_dir[1] * force * dt))

                # damage proportional to acceleration
                hittable.take_damage(force * dt)

class Hittable:
    """Base class for objects that can take damage"""
    def __init__(
            self, hitbox: Optional[pygame.Rect],
            position: Tuple[float, float],
            health: int,
            i_time: int = 600,
            hittables: List["Hittable"] = []):
        self.position = position
        self.velocity = (0.0, 0.0)
        self.health = health
        self.max_health = health
        self.hitbox = hitbox        # Simple square hitbox (do not draw)
        self.i_time = i_time        # Invincibility time after being hit in ms
        self.hittables = hittables
        self.invincible = False
        self.i_frames_start = -9999
    def take_damage(self, amount: int):
        if self.invincible:
            return
        self.health -= amount
        if self.health < 0:
            self.health = 0
        self.invincible = True
        self.i_frames_start = pygame.time.get_ticks()
    def pushed(self, force: Tuple[float, float]):
        self.velocity = (self.velocity[0] + force[0],
                         self.velocity[1] + force[1])
    def update(self):
        if self.invincible:
            current_time = pygame.time.get_ticks()
            if current_time - self.i_frames_start > self.i_time:
                self.invincible = False
        if self.is_destroyed():
            self.health = 0
            self.hitbox = None
    def is_destroyed(self) -> bool:
        return self.health <= 0

class Player(Hittable):
    """Player agent for Arena environment"""
    def __init__(self, position: Tuple[float, float],
                 angle: float = 0.0,
                 difficulty: int = 0):
        rect = pygame.Rect(position[0] - PLAYER_HITBOX_SIZE // 2,
                           position[1] - PLAYER_HITBOX_SIZE // 2,
                           PLAYER_HITBOX_SIZE,
                           PLAYER_HITBOX_SIZE)
        i_time = max(100, 600 - difficulty * 50)
        super().__init__(hitbox=rect, position=position, health=100, i_time=i_time)
        self.angle = angle
        self.power = 10
        self.max_speed = max(3.0, 5.0 - difficulty * 0.1)
        self.thrust = max(0.1, 0.2 - difficulty * 0.025)
        self.rotation_speed = 15.0 # degrees per action
    def rotate(self, rotation_direction: int):
        """Rotate player by rotation_direction (1 clockwise, -1 anti-clockwise)"""
        self.angle += self.rotation_speed * rotation_direction
    def thruster(self):
        """Move player forward in the direction they are facing"""
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.thrust
        dy = math.sin(rad) * self.thrust
        self.velocity = (self.velocity[0] + dx, self.velocity[1] + dy)
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
    def inertial_manipulator_up(self):
        """Move player up (negative y)"""
        self.angle = 270.0
        self.velocity = (self.velocity[0], self.velocity[1] - self.thrust)
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
    def inertial_manipulator_down(self):
        """Move player down (positive y)"""
        self.angle = 90.0
        self.velocity = (self.velocity[0], self.velocity[1] + self.thrust)
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
    def inertial_manipulator_left(self):
        """Move player left (negative x)"""
        self.angle = 180.0
        self.velocity = (self.velocity[0] - self.thrust, self.velocity[1])
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
    def inertial_manipulator_right(self):
        """Move player right (positive x)"""
        self.angle = 0.0
        self.velocity = (self.velocity[0] + self.thrust, self.velocity[1])
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
    def shoot(self) -> Bullet:
        """Create a bullet moving in the direction the player is facing"""
        rad = math.radians(self.angle)
        direction = (math.cos(rad), math.sin(rad))
        bullet_start_pos = (self.position[0] + direction[0] * 5,
                            self.position[1] + direction[1] * 5)
        return Bullet(bullet_start_pos, direction, damage=self.power, hittables=self.hittables)
    def heal(self, amount: int):
        """Heal the player by amount, increasing max health and damage if overheal"""
        # Polygonkind is dead
        # Area is fuel
        # GPU is full
        self.health += amount
        if self.health > self.max_health:
            overheal = self.health - self.max_health
            self.max_health += math.ceil(overheal / 10)
            self.health = self.max_health
            self.power += math.floor(overheal / 20)     # Harder to increase power
    def update(self):
        # Update invincibility frames
        super().update()
        # Update position
        self.position = (self.position[0] + self.velocity[0],
                         self.position[1] + self.velocity[1])
        # Update hitbox
        self.hitbox.update(self.position[0] - 10, self.position[1] - 10, 20, 20)
        # Apply friction
        self.velocity = (self.velocity[0] * 0.9, self.velocity[1] * 0.9)
        # Stop if velocity is very low
        if abs(self.velocity[0]) < 0.01:
            self.velocity = (0.0, self.velocity[1])
        if abs(self.velocity[1]) < 0.01:
            self.velocity = (self.velocity[0], 0.0)

class EnemyTypes(enum.Enum):
    RAMMER = 0
    TANKIER_RAMMER = 1
    EXPLOSIVE_RAMMER = 2
    GOTTAGOFAST = 3
    PEW_PEW = 4
    BIG_PEW_PEW = 5
    SPAWNCEPTION = 6
    DIFFICULTY_LONGINUS = 7

class LonginusMoveList(enum.Enum):
    PRESPELL_1 = 0
    SPELL_1 = 1
    PRESPELL_2 = 2
    SPELL_2 = 3
    PRESPELL_3 = 4
    SPELL_3 = 5
    LAST_WORD = 6

enemy_type_modifiers = {
    EnemyTypes.RAMMER: {"health": 100.0, "damage": 100.0, "speed": 100.0, "force": 100.0, "size": 100.0, "reward": 100.0},
    EnemyTypes.TANKIER_RAMMER: {"health": 150.0, "damage": 100.0, "speed": 80.0, "force": 80.0, "size": 250.0, "reward": 150.0},
    EnemyTypes.EXPLOSIVE_RAMMER: {"health": 50.0, "damage": 200.0, "speed": 100.0, "force": 120.0, "size": 150.0, "reward": 150.0},
    EnemyTypes.GOTTAGOFAST: {"health": 0.01, "damage": 100.0, "speed": 150.0, "force": 10000.0, "size": 100.0, "reward": 120.0},
    EnemyTypes.PEW_PEW: {"health": 80.0, "damage": 100.0, "speed": 90.0, "force": 90.0, "size": 100.0, "reward": 140.0},
    EnemyTypes.BIG_PEW_PEW: {"health": 120.0, "damage": 150.0, "speed": 80.0, "force": 80.0, "size": 200.0, "reward": 180.0},
    EnemyTypes.SPAWNCEPTION: {"health": 500.0, "damage": 0.0, "speed": 10.0, "force": 10.0, "size": 500.0, "reward": 300.0},
    EnemyTypes.DIFFICULTY_LONGINUS: {"health": 7000.0, "damage": float('inf'), "speed": 100.0, "force": float('inf'), "size": 1000.0, "reward": 1000000.0},
}

class Enemy(Hittable):
    """Enemy object"""
    def __init__(self, position: Tuple[float, float],
                 difficulty: int = 0,
                 type: EnemyTypes = EnemyTypes.RAMMER,
                 target: Optional["Player"] = None):
        this_size = int(round(ENEMY_HITBOX_SIZE * (enemy_type_modifiers[type]["size"] / 100.0)))
        rect = pygame.Rect(position[0] - this_size / 2,
                           position[1] - this_size / 2,
                           this_size,
                           this_size)
        health = int(round((5 + difficulty * 5) * (enemy_type_modifiers[type]["health"] / 100.0)))
        damage = int(round((1 + difficulty * 1) * (enemy_type_modifiers[type]["damage"] / 100.0)))
        max_speed = (2.0 + difficulty / 20.0) * (enemy_type_modifiers[type]["speed"] / 100.0)
        force = (0.05 + difficulty * 0.01) * (enemy_type_modifiers[type]["force"] / 100.0)
        super().__init__(hitbox=rect, position=position, health=health)
        self.difficulty = difficulty
        self.target = target
        self.type = type
        self.damage = damage
        self.max_speed = max_speed
        self.force = force
        self.reward = int(round(difficulty * (enemy_type_modifiers[type]["reward"] / 100.0)))
        self.timing_help = pygame.time.get_ticks()
    def move(self, accel: Tuple[float, float]):
        self.velocity = (self.velocity[0] + accel[0],
                         self.velocity[1] + accel[1])
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
        self.position = (self.position[0] + self.velocity[0],
                         self.position[1] + self.velocity[1])
        self.hitbox.update(self.position[0] - self.hitbox.width / 2,
                           self.position[1] - self.hitbox.height / 2,
                           self.hitbox.width,
                           self.hitbox.height)
    def shoot(self) -> Optional[Bullet]:
        """Create a bullet moving towards the player"""
        if self.target is None:
            return None
        to_player = vectorHelper.vec_sub(self.target.position, self.position)
        direction = vectorHelper.vec_norm(to_player)
        bullet_start_pos = (self.position[0] + direction[0] * (self.hitbox.width / 2 + 5),
                            self.position[1] + direction[1] * (self.hitbox.height / 2 + 5))
        return Bullet(bullet_start_pos, direction, owner=self, damage=self.damage, hittables=self.hittables)
    def explode(self) -> Explosion:
        """Create an explosion at the enemy's position"""
        return Explosion(position=self.position, owner=self, damage=self.damage, hittables=self.hittables, radius=self.hitbox.width)
    def achieve_goal(self, target: "Player"):
        """Finite State Machines, depending on enemy type"""
        if self.type in {EnemyTypes.RAMMER, EnemyTypes.TANKIER_RAMMER, EnemyTypes.EXPLOSIVE_RAMMER, EnemyTypes.GOTTAGOFAST}:
            # Move towards player
            goal = target.position + target.velocity * min(1, self.difficulty * 0.05) # Predictive targeting increases with difficulty
            direction = vectorHelper.vec_sub(goal, self.position)
            direction = vectorHelper.vec_norm(direction)
            acceleration = (direction[0] * self.force, direction[1] * self.force)
            self.move(acceleration)
            # Explode if close enough (Explosive Rammer only)
            if self.type == EnemyTypes.EXPLOSIVE_RAMMER:
                distance = vectorHelper.vec_len(vectorHelper.vec_sub(target.position, self.position))
                if distance <= (self.hitbox.width + target.hitbox.width / 2):
                    
                    self.health = 0  # Self-destruct
        elif self.type in {EnemyTypes.PEW_PEW, EnemyTypes.BIG_PEW_PEW}:
            # Firing range
            aim_range = 300 + self.difficulty * 10
            to_player = vectorHelper.vec_sub(target.position, self.position)
            distance = vectorHelper.vec_mag(to_player)
            direction = vectorHelper.vec_norm(to_player)
            if distance <= aim_range:
                # Shoot at player
                current_time = pygame.time.get_ticks()
                shoot_interval = max(500 - self.difficulty * 20, 200)
                if current_time - self.timing_help >= shoot_interval:
                    self.timing_help = current_time
                    return Bullet(
                        position=(self.position[0] + direction[0] * (self.hitbox.width / 2 + 5),
                                  self.position[1] + direction[1] * (self.hitbox.height / 2 + 5)),
                        direction=direction,
                        damage=self.damage,
                        hittables={target}
                    )
            else:
                # Move closer to player
                acceleration = (direction[0] * self.force, direction[1] * self.force)
                self.move(acceleration)


    def update(self):
        super().update()
        self.achieve_goal(self, self.target)  # Placeholder to avoid error

class Spawner(Hittable):
    """Enemy spawner object"""
    def __init__(self, position: Tuple[float, float],
                 difficulty: int = 0):
        rect = pygame.Rect(position[0] - SPAWNER_HITBOX_SIZE // 2,
                           position[1] - SPAWNER_HITBOX_SIZE // 2,
                           SPAWNER_HITBOX_SIZE,
                           SPAWNER_HITBOX_SIZE)
        health = 100 + difficulty * 10
        super().__init__(hitbox=rect, position=position, health=health)
        self.spawn_timer = max(500, 5000 - difficulty * 200) # in ms
        self.spawn_level = math.floor(math.cbrt(random.randint(0, difficulty ** 2)))
        self.last_spawn_time = random.randint(0, self.spawn_timer)
    def update(self):
        super().update()
