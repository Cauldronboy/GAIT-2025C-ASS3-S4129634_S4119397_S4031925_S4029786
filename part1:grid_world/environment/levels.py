"""
Level layouts for the GridWorld RL project.

Legend:
    S = Start position
    A = Apple (+1 reward)
    K = Key (no reward, needed for chests)
    C = Chest (+2 reward, requires key)
    R = Rock (blocks movement)
    F = Fire (instant death)
    M = Monster (instant death if collision)
    . = Empty space
"""

# Level 0: Basic apples collection - shortest path learning
LEVEL_0 = [
    "S           ",
    "            ",
    "        A   ",
    "        A   ",
    "        A   ",
    "        A   ",
    "        A   ",
    "            "
]

# Level 1: Apples with fire hazards - SARSA should be more conservative
LEVEL_1 = [
    "S           ",
    "  F   F     ",
    "  F   F A   ",
    "        A   ",
    "  F   F A   ",
    "  F   F     ",
    "            ",
    "            "
]

# Level 2: Multiple apples, key, and chest
LEVEL_2 = [
    "S     R     ",
    "      R     ",
    "A     R   K ",
    "A     R     ",
    "A     R   C ",
    "      R     ",
    "            ",
    "            "
]

# Level 3: More complex key-chest puzzle
LEVEL_3 = [
    "S  R  R  R  ",
    "   R     R  ",
    "A  R  K  R A",
    "A  R     R A",
    "   R  C  R  ",
    "   R  R  R  ",
    "            ",
    "      A     "
]

# Level 4: Monsters with simple movement
LEVEL_4 = [
    "S     R     ",
    "  M   R     ",
    "      R   A ",
    "  M       A ",
    "      R   A ",
    "  M   R     ",
    "            ",
    "            "
]

# Level 5: Multiple monsters with key-chest objective
LEVEL_5 = [
    "S  R  M  R  ",
    "   R     R  ",
    "A  R  K  R A",
    "M           ",
    "   R  C  R M",
    "   R  M  R  ",
    "            ",
    "      A     "
]

# Level 6: Complex layout for intrinsic reward exploration
LEVEL_6 = [
    "S  R  R  R  ",
    "   R     R  ",
    "A  R     R A",
    "         K  ",
    "A  R  C  R A",
    "   R     R  ",
    "            ",
    "      A     "
]

# Map level index to layout
LEVELS = {
    0: LEVEL_0,
    1: LEVEL_1,
    2: LEVEL_2,
    3: LEVEL_3,
    4: LEVEL_4,
    5: LEVEL_5,
    6: LEVEL_6
}

def get_level(level_num: int):
    """Get level layout by number."""
    if level_num not in LEVELS:
        raise ValueError(f"Level {level_num} not found. Available levels: {list(LEVELS.keys())}")
    
    # Ensure all rows have the same width
    layout = LEVELS[level_num]
    max_width = max(len(row) for row in layout)
    return [row.ljust(max_width)[:max_width] for row in layout]

def get_level_name(level_num: int) -> str:
    """Get descriptive name for level."""
    names = {
        0: "Level 0: Basic Apples",
        1: "Level 1: Fires & Hazards",
        2: "Level 2: Keys & Chests",
        3: "Level 3: Complex Puzzle",
        4: "Level 4: Monster Patrol",
        5: "Level 5: Monster Challenge",
        6: "Level 6: Intrinsic Exploration"
    }
    return names.get(level_num, f"Level {level_num}")
