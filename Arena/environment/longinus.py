"""Longinus gets its own file because of the complexity"""
from .entities import Bullet, Enemy, EnemyTypes
from typing import Tuple, Dict, List, Optional, Set
import random
import enum


class LonginusPhaseList(enum.Enum):
    PRESPELL_1 = 0
    SPELL_1 = 1
    PRESPELL_2 = 2
    SPELL_2 = 3
    PRESPELL_3 = 4
    SPELL_3 = 5
    LAST_WORD = 6


class Danmaku(Bullet):
    """Bullet with colors"""
    def __init__(self, position,
                 direction,
                 owner = None,
                 damage = 10,
                 speed = 20,
                 size = 1,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 env = None):
        super().__init__(position, direction, owner, damage, speed, size, env)
        self.color = color

def encode_danmaku_bullet(distance_from_center: float, angle_from_aim: float, speed: int = 20,
                          size: int = 1, delay: int = 0, color: Tuple[int, int, int] = (255, 255, 255)) -> Dict[str, float]:
    """Encode a Danmaku bullet for a shot"""
    danmaku_bullet = {"distance from shooter": distance_from_center,
               "angle from aim": angle_from_aim,
               "speed": float(speed), "size": float(size), "delay": float(delay),
               "red": float(color[0]), "green": float(color[1]), "blue": float(color[2])}
    return danmaku_bullet

class BulletSpeedScaleStyle(enum.Enum):
    DISTANCE_FROM_ORIGIN = 0
    BURST_INDEX = 1
    BULLET_INDEX_IN_LINE = 2
    RANDOM = 3

class OffsetStyle(enum.Enum):
    EVENLY_SPACED = 0
    EXPLICIT = 1
    RANDOM = 2

# TODO: fix customization options
def lines_shot_builder(target_list: List[Dict[str, float]],
                       burst_amount: int = 1, burst_delay: List[int] = [0],
                       shot_start: Optional[int] = None, shot_duration: Optional[int] = None,
                       burst_line_amount: List[int] = [1],
                       burst_line_angle_offset_max: float = 0.0,
                       burst_line_angle_offset_style: List[OffsetStyle] = OffsetStyle.EVENLY_SPACED,
                       burst_line_angle_offset: Optional[List[List[float]]] = None,
                       burst_line_bullet_amount: int = 1, burst_line_bullet_dist: List[float] = [4.0],
                       randomize_bullet_offset: bool = False, max_bullet_offset: float = 0.0,
                       default_bullet_speed: int = 20, bullet_speed_scale_style: Set[BulletSpeedScaleStyle] = [],
                       bullet_speed_scale_speed: Tuple[float, float, float] = (1.0, 1.0, 1.0, 1.0)):
    """
    Generate and append encoded danmaku bullets to provided target list\n
        Warning: if burst_amount is lower than the length of burst_delay and\n
        burst_line_amount lists, higher indeces will be skipped\n
        However, if burst_amount is higher than said lengths, the list will\n
        wrap around
    """
    # Interrupt if no target list is provided
    if target_list is None:
        return
    # If shot duration is provided and burst delay list is empty, calculate burst_delay
    if shot_duration is not None and len(burst_delay) == 0:
        burst_delay.clear()
        for i in range(burst_amount):
            delay = i * int(shot_duration / burst_amount)
            burst_delay.append(delay)
    

    for burst_no in range(burst_amount):
        delay = burst_delay[burst_no % len(burst_delay)]
        line_amount = burst_line_amount[burst_no % len(burst_line_amount)]
        for line in range(line_amount):
            angle = burst_line_angle_offset_max * line / (line_amount - 1) - burst_line_angle_offset_max / 2
            for dist in burst_line_bullet_dist:
                # Bullet speed scaling
                speed = default_bullet_speed
                speed *= dist * bullet_speed_scale_speed[0] if BulletSpeedScaleStyle.DISTANCE_FROM_ORIGIN in bullet_speed_scale_style else 1
                speed *= burst_no * bullet_speed_scale_speed[1] if BulletSpeedScaleStyle.BURST_INDEX in bullet_speed_scale_style else 1
                speed = speed ** (random.random() * 2 * bullet_speed_scale_speed[3]) if BulletSpeedScaleStyle.RANDOM in bullet_speed_scale_style else speed
                color_r = int(burst_no * 127.5)
                color_g = int((2 - burst_no) * 127.5)
                target_list.append[encode_danmaku_bullet(dist, angle, speed, 2, delay, (color_r, color_g, 0))]
    

class Longinus(Enemy):
    """
    Bossfight
    TODO: moveset
    """
    def __init__(self, position,
                 difficulty = 0,
                 type = EnemyTypes.DIFFICULTY_LONGINUS,
                 target = None,
                 iteration = None,
                 env = None):
        super().__init__(position, difficulty, type, target, iteration, env)
        self.phase = LonginusPhaseList.PRESPELL_1
        self.pattern: List[List[Dict[str, float]]] = []
        
    def achieve_goal(self, current_time):

        return

    def prespell_1(self):
        """
        Move to random location\n
        - Option 1: Shoot 3 bursts of 3, 5, 7 lines of 8 bullets at target, each faster than the previous\n
        - Option 2: Shoot 4 big bullets in 4 cardinal directions, then another 4 diagonally
        """
        ps1_sg_burst_line_bullet_dist = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]  # Distance from origin 
        ps1_sg_burst_line_spread_angle = 105.0                                          # Total spread angle
        ps1_sg_burst_line_amount = [3, 5, 7]                                            # Number of lines in a burst
        ps1_sg_burst_delay = [0, 400, 800]                                              # Delay of burst, length is amount of bursts
        shotgun = []                                # Shot container