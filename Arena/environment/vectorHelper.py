"""Helper functions for coordinate-based math"""

import math
from typing import Tuple


# Vector helpers
def vec_norm(v: Tuple[float, float]) -> Tuple[float, float]:
    """Return vector of length 1 pointing at the same direction"""
    length = math.hypot(v[0], v[1])
    if length == 0:
        return (0.0, 0.0)
    
    # Handle infinite values
    if not math.isfinite(v[0]) or not math.isfinite(v[1]):
        sx = 0.0 if v[0] == 0 else math.copysign(1.0, v[0])
        sy = 0.0 if v[1] == 0 else math.copysign(1.0, v[1])

        length = math.hypot(sx, sy)
        if length == 0.0:
            return (0.0, 0.0)

        return (sx / length, sy / length)

    return (v[0] / length, v[1] / length)

def dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Return dot product of 2 vectors"""
    return v1[0] * v2[0] + v1[1] * v2[1]

def vec_len(*args) -> float:
    """
    (Tuple): Vector length\n
    (List[Tuple[]]): Distance between first and last point in list\n
    (Tuple, Tuple): Distance between two points
    """
    if len(args) == 1:
        x = args[0]

        # Vector length: (x, y)
        if isinstance(x, tuple) and len(x) == 2:
            return math.hypot(x[0], x[1])

        # Segment length: [(x1, y1), ...ignored..., (x2, y2)]
        if isinstance(x, list) and len(x) >= 2:
            p1, p2 = x[0], x[-1]
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    elif len(args) == 2:
        # Distance between two points
        p1, p2 = args
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    raise TypeError("Invalid arguments for length()")

def vec_to_ang(v: Tuple[float, float]) -> float:
    """Angle from origin (degrees)"""
    return math.degrees(math.atan2(v[1], v[0])) % 360

def ang_to_vec(d: float) -> Tuple[float, float]:
    """Vector from angle (degrees)"""
    x = math.cos(math.radians(d))
    y = math.sin(math.radians(d))
    return (x, y)

def vec_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple:
    """Subtract vector b from a."""
    # Used to find the difference between two points or velocities.
    return (a[0]-b[0], a[1]-b[1])

def vec_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple:
    """Add two vectors together."""
    # Used to combine movement or steering effects.
    return (a[0]+b[0], a[1]+b[1])

def vec_mul(v: Tuple[float, float], scalar: float) -> Tuple:
    """Multiply a vector by a scalar number."""
    # This scales a direction vector to a desired speed or force magnitude.
    return (v[0]*scalar, v[1]*scalar)

def vec_lim(v: Tuple[float, float], max_value: float) -> Tuple:
    """Limit the magnitude (length) of a vector to a maximum value."""
    length = vec_len(v)
    if length > max_value:
        # If it’s too long, shrink it back to the maximum allowed length.
        v = vec_norm(v)
        return (v[0]*max_value, v[1]*max_value)
    # Otherwise return it unchanged.
    return v

def vec_rotate(v: Tuple[float, float], theta: float) -> Tuple:
    """Rotate the vector by angle, positive is clockwise"""
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    vx = v[0]
    vy = v[1]
    return (vx*cos_theta - vy*sin_theta, vx*sin_theta + vy*cos_theta)