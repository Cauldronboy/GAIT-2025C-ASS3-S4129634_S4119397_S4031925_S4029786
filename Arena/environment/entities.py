"""Entities in the Arena environment."""

import pygame
import math
import random
import enum
from typing import Tuple, List, Set, Dict, Optional
import vectorHelper
from arena import hittables, bullets, ARENA_CORNERS, ARENA_WIDTH, ARENA_HEIGHT


# Object hitbox sizes
PLAYER_HITBOX_SIZE = 20
BULLET_HITBOX_SIZE = 4
ENEMY_HITBOX_SIZE = 20
SPAWNER_HITBOX_SIZE = 40


# Global variables
DRAG = 0.1      # How hard acceleration moves against velocity


# Shape sweep collision detection
def rect_sweep(
    rect: pygame.Rect,
    vel: Tuple[float, float],
    obstacles: List["Hittable"],
    exceptions: Set["Hittable"] = None
) -> Optional["Hittable"]:
    """
    Swept AABB collision detection.
    Assumes rect is at its starting position.
    Returns the first Hittable hit, or None.
    """

    if exceptions is None:
        exceptions = set()

    vx, vy = vel
    earliest_t = 1.0
    hit_object = None

    not_friendlies = [obs for obs in obstacles if obs not in exceptions
                      and obs.hitbox is not None]

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
                 speed: int = 20):
        self.owner = owner
        self.position = position
        self.direction = vectorHelper.vec_norm(direction)
        self.speed = speed
        self.damage = damage
        self.hitbox = pygame.Rect(position[0] - BULLET_HITBOX_SIZE // 2,
                                  position[1] - BULLET_HITBOX_SIZE // 2,
                                  BULLET_HITBOX_SIZE,
                                  BULLET_HITBOX_SIZE)
        bullets.append(self)
    def hit(self, hittable: "Hittable"):
        hittable.take_damage(self.damage)
        bullets.remove(self)
    def update(self, dt: float):
        """Update object depending on the time since last update in seconds"""
        if self not in bullets:
            return
        movement = (self.direction[0] * self.speed * dt, self.direction[1] * self.speed * dt)
        gottem = rect_sweep(self.hitbox, movement, hittables, exceptions={self.owner} if self.owner else None)
        if gottem is not None:
            self.hit(gottem)
            return
        self.position = (self.position[0] + movement[0], self.position[1] + movement[1])
        # If fully out of bounds, remove self
        if (
            self.position[0] < 0 - self.hitbox.width / 2 or
            self.position[0] > ARENA_WIDTH + self.hitbox.width / 2 or
            self.position[1] < 0 - self.hitbox.height / 2 or
            self.position[1] > ARENA_HEIGHT + self.hitbox.height / 2
           ):
            bullets.remove(self)
        self.hitbox.update(int(self.position[0] - self.hitbox.width / 2),
                           int(self.position[1] - self.hitbox.height / 2),
                           self.hitbox.width,
                           self.hitbox.height)

class Explosion(Bullet):
    """Basically a stationary bullet that damages on contact"""
    def __init__(self, position: Tuple[float, float],
                 owner: Optional["Hittable"] = None,
                 damage: int = 50,
                 radius: int = 20):
        super().__init__(position, (0, 0), owner, damage, 0)
        self.hitbox = pygame.Rect(position[0] - radius,
                                  position[1] - radius,
                                  radius * 2,
                                  radius * 2)
        self.life_expectancy = 500    # milliseconds
        self.start_time = pygame.time.get_ticks()
        self.radius = radius
    def update(self, dt: float):
        """Update object depending on the time since last update in seconds"""
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time
        t = max(0.0, 1.0 - elapsed / self.life_expectancy)
        remaining_damage = self.damage * t
        if current_time - self.start_time >= self.life_expectancy:
            bullets.remove(self)
            return
        feasably_hit = [h for h in hittables if h.hitbox
                        and vectorHelper.vec_len(self.position, h.position) <= self.radius + math.hypot(h.hitbox.width/2, h.hitbox.width/2)]
        for hittable in feasably_hit:
            if hittable.hitbox and self.hitbox.colliderect(hittable.hitbox):

                hit_dir = vectorHelper.vec_sub(hittable.position, self.position)
                hit_dir = vectorHelper.vec_norm(hit_dir)
                
                dist = max(1e-6, vectorHelper.vec_len(self.position, hittable.position))
                falloff = (self.radius / 2) / ((dist + 1.0) ** 2)

                force = remaining_damage * falloff

                # knockback = impulse
                hittable.pushed((hit_dir[0] * force * dt,
                                hit_dir[1] * force * dt))

                # damage proportional to acceleration
                hittable.take_damage(int(round(force * dt)))

class Hittable:
    """Base class for objects that can take damage"""
    def __init__(
            self, position: Tuple[float, float],
            health: int,
            max_speed: float = 0,
            hitbox: Optional[pygame.Rect] = None,
            i_time: int = 600):
        self.position = position
        self.velocity = (0.0, 0.0)
        self.accel = (0.0, 0.0)
        self.max_speed = max_speed
        self.health = health
        self.max_health = health
        if hitbox is None:
            hitbox = pygame.Rect(self.position[0] - 10,
                                 self.position[1] - 10,
                                 20, 20)
        self.hitbox = hitbox        # Simple square hitbox (do not draw)
        self.i_time = i_time        # Invincibility time after being hit in ms
        self.invincible = False
        self.i_frames_start = -9999
        # Register self in hittables list
        hittables.append(self)
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
    def update(self, dt: float):
        """Update object depending on the time since last update in seconds"""
        if self.out_of_health():
            self.destroy()
        if self.invincible:
            current_time = pygame.time.get_ticks()
            if current_time - self.i_frames_start > self.i_time:
                self.invincible = False
        # Update acceleration
        self.accel = vectorHelper.vec_add(self.accel, vectorHelper.vec_mul(self.velocity, -DRAG))
        # Update velocity
        self.velocity = (self.velocity[0] + self.accel[0] * dt,
                         self.velocity[1] + self.accel[1] * dt)
        self.velocity = vectorHelper.vec_lim(self.velocity, self.max_speed)
        # Update position
        self.position = (self.position[0] + self.velocity[0] * dt,
                         self.position[1] + self.velocity[1] * dt)
        # If fully out of bounds, remove self
        if (
            self.position[0] < 0 - self.hitbox.width / 2 or
            self.position[0] > ARENA_WIDTH + self.hitbox.width / 2 or
            self.position[1] < 0 - self.hitbox.height / 2 or
            self.position[1] > ARENA_HEIGHT + self.hitbox.height / 2
           ):
            if isinstance(self, Enemy) and self.type == EnemyTypes.EXPLOSIVE_RAMMER:
                # Explode before being removed
                self.explode()
            self.destroy()
        # Stop if velocity is very low
        if abs(self.velocity[0]) < 0.01:
            self.velocity = (0.0, self.velocity[1])
        if abs(self.velocity[1]) < 0.01:
            self.velocity = (self.velocity[0], 0.0)

        if self.hitbox is not None:
            # Update hitbox if not destroyed
            self.hitbox.update(int(self.position[0] - self.hitbox.width / 2),
                               int(self.position[1] - self.hitbox.height / 2),
                               self.hitbox.width,
                               self.hitbox.height)
        # Reset acceleration for next frame
        self.accel = (0.0, 0.0)

    def out_of_health(self) -> bool:
        """Returns true if health is at or below 0 but not negative infinity"""
        return self.health <= 0 and self.health > float('-inf')
    def destroy(self):
        """Remove self from hittables list"""
        if self in hittables:
            hittables.remove(self)

class Player(Hittable):
    """Player agent for Arena environment"""
    def __init__(self, position: Tuple[float, float],
                 angle: float = 0.0):
        rect = pygame.Rect(position[0] - PLAYER_HITBOX_SIZE // 2,
                           position[1] - PLAYER_HITBOX_SIZE // 2,
                           PLAYER_HITBOX_SIZE,
                           PLAYER_HITBOX_SIZE)
        super().__init__(position=position, health=100, max_speed=5.0, hitbox=rect, i_time=600)
        self.angle = angle
        self.power = 10
        self.thrust = 0.2
        self.rotation_speed = 15.0 # degrees per action
    def rotate(self, rotation_direction: int):
        """Rotate player by rotation_direction (1 clockwise, -1 anti-clockwise)"""
        self.angle += self.rotation_speed * rotation_direction
    def activate_thrust(self):
        """Move player forward in the direction they are facing"""
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.thrust
        dy = math.sin(rad) * self.thrust
        self.accel = (dx, dy)
    def inertial_manipulator_up(self):
        """Move player up (negative y)"""
        self.angle = 270.0
        self.activate_thrust()
    def inertial_manipulator_down(self):
        """Move player down (positive y)"""
        self.angle = 90.0
        self.activate_thrust()
    def inertial_manipulator_left(self):
        """Move player left (negative x)"""
        self.angle = 180.0
        self.activate_thrust()
    def inertial_manipulator_right(self):
        """Move player right (positive x)"""
        self.angle = 0.0
        self.activate_thrust()
    def shoot(self) -> Bullet:
        """Create a bullet moving in the direction the player is facing"""
        rad = math.radians(self.angle)
        direction = (math.cos(rad), math.sin(rad))
        bullet_start_pos = (self.position[0] + direction[0] * 5,
                            self.position[1] + direction[1] * 5)
        return Bullet(bullet_start_pos, direction, damage=self.power)
    def heal(self, amount: int):
        """Heal the player by amount, increasing max health and damage if overheal"""
        # Polygonkind is dead
        # Area is fuel
        # GPU is full
        self.health += amount
        if self.health > self.max_health:
            overheal = self.health - self.max_health
            self.max_health += int(overheal / 10)
            self.health = self.max_health
            self.power += int(overheal / 20)        # Harder to increase power
    def update(self, dt: float):
        # Update invincibility frames
        super().update(dt)

class EnemyTypes(enum.Enum):
    RAMMER = 0
    TANKIER_RAMMER = 1
    EXPLOSIVE_RAMMER = 2
    GOTTAGOFAST = 3
    PEW_PEW = 4
    BIG_PEW_PEW = 5
    SPAWNCEPTION = 6
    DIFFICULTY_LONGINUS = 7

class LonginusPhaseList(enum.Enum):
    PRESPELL_1 = 0
    SPELL_1 = 1
    PRESPELL_2 = 2
    SPELL_2 = 3
    PRESPELL_3 = 4
    SPELL_3 = 5
    LAST_WORD = 6

enemy_type_modifiers = {
    EnemyTypes.RAMMER: {"health": 100.0, "damage": 100.0, "speed": 100.0, "force": 100.0, "size": 100.0, "cooldown": 0.0, "reward": 100.0},
    EnemyTypes.TANKIER_RAMMER: {"health": 150.0, "damage": 100.0, "speed": 80.0, "force": 80.0, "size": 250.0, "cooldown": 0.0, "reward": 150.0},
    EnemyTypes.EXPLOSIVE_RAMMER: {"health": 50.0, "damage": 200.0, "speed": 100.0, "force": 120.0, "size": 150.0, "cooldown": 0.0, "reward": 150.0},
    EnemyTypes.GOTTAGOFAST: {"health": 0.01, "damage": 100.0, "speed": 150.0, "force": 10000.0, "size": 100.0, "cooldown": 0.0, "reward": 120.0},
    EnemyTypes.PEW_PEW: {"health": 80.0, "damage": 100.0, "speed": 90.0, "force": 90.0, "size": 100.0, "cooldown": 100.0, "reward": 140.0},
    EnemyTypes.BIG_PEW_PEW: {"health": 120.0, "damage": 150.0, "speed": 80.0, "force": 80.0, "size": 200.0, "cooldown": 120.0, "reward": 180.0},
    EnemyTypes.SPAWNCEPTION: {"health": 500.0, "damage": 0.0, "speed": 10.0, "force": 10.0, "size": 500.0, "cooldown": 1000.0, "reward": 300.0},
    EnemyTypes.DIFFICULTY_LONGINUS: {"health": 7000.0, "damage": float('inf'), "speed": 100.0, "force": float('inf'), "size": 800.0, "cooldown": 0.0, "reward": 1000000.0},
}

SPAWNCEPTION_MAX_ITERATION = 0

class Enemy(Hittable):
    """Enemy object"""
    def __init__(self, position: Tuple[float, float],
                 difficulty: int = 0,
                 type: EnemyTypes = EnemyTypes.RAMMER,
                 target: Optional["Player"] = None,
                 iteration: Optional[int] = None):
        this_size = int(round(ENEMY_HITBOX_SIZE * (enemy_type_modifiers[type]["size"] / 100.0)))
        rect = pygame.Rect(position[0] - this_size / 2,
                           position[1] - this_size / 2,
                           this_size,
                           this_size)
        health = int(round((5 + difficulty * 5) * (enemy_type_modifiers[type]["health"] / 100.0)))
        damage = int(round((1 + difficulty * 1) * (enemy_type_modifiers[type]["damage"] / 100.0)))
        max_speed = (2.0 + difficulty / 20.0) * (enemy_type_modifiers[type]["speed"] / 100.0)
        force = (0.05 + difficulty * 0.01) * (enemy_type_modifiers[type]["force"] / 100.0)
        super().__init__(position=position, health=health, max_speed=max_speed, hitbox=rect)
        self.difficulty = difficulty
        self.target = target                            # Player
        self.goal: Tuple[float, float] = (0.0, 0.0)     # Positional goal
        self.type = type
        self.damage = damage
        self.force = force
        self.reward = int(round(difficulty * (enemy_type_modifiers[type]["reward"] / 100.0)))
        self.last_cooldownable_action = -999
        self.cooldown = int(round(max(500, 1000 - difficulty * 500) * (enemy_type_modifiers[type]["cooldown"] / 100.0)))
        if self.type == EnemyTypes.SPAWNCEPTION:
            self.iteration = iteration if iteration is not None else 0
    def shoot(self) -> Optional[Bullet]:
        """Create a bullet moving towards the player"""
        if self.target is None:
            return None
        to_player = vectorHelper.vec_sub(self.target.position, self.position)
        direction = vectorHelper.vec_norm(to_player)
        bullet_start_pos = (self.position[0] + direction[0] * (self.hitbox.width / 2 + 5),
                            self.position[1] + direction[1] * (self.hitbox.height / 2 + 5))
        return Bullet(bullet_start_pos, direction, owner=self, damage=self.damage)
    def explode(self) -> Explosion:
        """Create an explosion at the enemy's position (self-destructing)"""
        self.health = float('-inf')
        return Explosion(position=self.position, owner=None, damage=self.damage, radius=self.hitbox.width)
    def achieve_goal(self, current_time):
        direction = vectorHelper.vec_sub(self.goal, self.position)
        distance = vectorHelper.vec_len(direction)
        direction = vectorHelper.vec_norm(direction)
        if self.type in {EnemyTypes.RAMMER, EnemyTypes.TANKIER_RAMMER, EnemyTypes.EXPLOSIVE_RAMMER, EnemyTypes.GOTTAGOFAST, EnemyTypes.SPAWNCEPTION}:
            self.accel = (direction[0] * self.force, direction[1] * self.force)
            # Explode if close enough (Explosive Rammer only)
            if self.type == EnemyTypes.EXPLOSIVE_RAMMER:
                distance = vectorHelper.vec_len(vectorHelper.vec_sub(self.target.position, self.position))
                if distance <= (self.hitbox.width + self.target.hitbox.width / 2):
                    self.explode()
        elif self.type in {EnemyTypes.PEW_PEW, EnemyTypes.BIG_PEW_PEW}:
            # Firing range
            aim_range = 300 + self.difficulty * 10
            if distance <= aim_range and current_time - self.last_cooldownable_action >= self.cooldown:
                # Shoot at player
                self.shoot()
                self.last_cooldownable_action = pygame.time.get_ticks()
            else:
                # Move closer to player
                self.accel = (direction[0] * self.force, direction[1] * self.force)
        # Extra feature of Spawnception
        if self.type == EnemyTypes.SPAWNCEPTION:
            # Scale max iterations based on difficulty
            actual_max_iter = int(round((SPAWNCEPTION_MAX_ITERATION + self.difficulty / 10)))
            # Check cooldown
            if current_time - self.last_cooldownable_action >= self.cooldown:
                self.last_cooldownable_action = pygame.time.get_ticks()
                # Spawn another Spawnception if current iteration is less than actual max iteration
                if self.iteration < actual_max_iter:
                    # Weaker anti proportional to current iteration
                    Enemy(position=self.position, difficulty=max(0, self.difficulty-int(self.difficulty*(self.iteration/float(actual_max_iter)))),
                        type=self.type, target=self.target, iteration=self.iteration+1)
                else:
                    Enemy(position=self.position, difficulty=self.difficulty,
                          type=EnemyTypes.RAMMER, target=self.target)


    def find_goal(self):
        """FSM, depending on enemy type"""
        if self.type in {EnemyTypes.RAMMER, EnemyTypes.TANKIER_RAMMER, EnemyTypes.EXPLOSIVE_RAMMER, EnemyTypes.GOTTAGOFAST,
                         EnemyTypes.PEW_PEW, EnemyTypes.BIG_PEW_PEW}:
            # Move towards player
            self.goal = vectorHelper.vec_add(self.target.position,          # Predictive targeting increases with difficulty
                                             vectorHelper.vec_mul(self.target.velocity,
                                                                  min(1, self.difficulty * 0.05)))
        elif self.type == EnemyTypes.SPAWNCEPTION:
            # Drift toward the furthest corner of the screen, spawning Rammers periodically
            self.goal: Tuple = None
            longest_dist = 0
            for corner in ARENA_CORNERS:
                temp_goal = ARENA_CORNERS[corner]
                dist = vectorHelper.vec_len(self.position, temp_goal)
                if dist > longest_dist:
                    longest_dist = dist
                    self.goal = temp_goal
        elif self.type == EnemyTypes.DIFFICULTY_LONGINUS:
            # How did you get here???
            self.goal = ( 999999.0, 999999.0)
            print("Wrong class")
    def update(self, dt: float):
        # If Explosive Rammers run out of health, they explode
        if self.out_of_health() and self.type == EnemyTypes.EXPLOSIVE_RAMMER:
            self.explode()
        if self.target is not None:
            self.find_goal()
        super().update(dt)

class Spawner(Hittable):
    """Enemy spawner object"""
    def __init__(self, position: Tuple[float, float],
                 difficulty: int = 0):
        rect = pygame.Rect(position[0] - SPAWNER_HITBOX_SIZE // 2,
                           position[1] - SPAWNER_HITBOX_SIZE // 2,
                           SPAWNER_HITBOX_SIZE,
                           SPAWNER_HITBOX_SIZE)
        health = 100 + difficulty * 10
        super().__init__(position=position, health=health, hitbox=rect)
        self.spawn_timer = max(500, 5000 - difficulty * 200) # in ms
        self.spawn_level = math.floor(math.cbrt(random.randint(0, difficulty ** 2)))
        self.last_spawn_time = random.randint(0, self.spawn_timer)
        self.source = Fabricator()
    def update(self, dt: float):
        super().update(dt)
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time >= self.spawn_timer:
            self.source.spawn_enemy(self.position)
            self.last_spawn_time = current_time

class Fabricator:
    def spawn_enemy(self, position: Tuple[float, float], type: EnemyTypes = EnemyTypes.RAMMER):
        """
        Spawn enemy appropriate to level
        TODO: actually spawn enemy
        TODO: despawn immediately after spawning a Longinus
        """
        pass