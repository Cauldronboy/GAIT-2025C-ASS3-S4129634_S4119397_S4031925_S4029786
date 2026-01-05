"""
Core GridWorld environment for reinforcement learning.

Supports:
- Movement in 4 directions with rock blocking
- Apples, keys, chests
- Fire hazards and monsters
- Probabilistic monster movement
- State encoding with apple/chest collection tracking
"""

import random
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, Optional


# Actions: 0=up, 1=right, 2=down, 3=left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]


@dataclass
class StepResult:
    """Result of taking an action in the environment."""
    next_state: Tuple  # State representation
    reward: float
    done: bool
    info: dict


class GridWorld:
    """
    GridWorld environment with full mechanics support.
    
    State representation:
    - For levels without monsters: (x, y, apple_mask, has_key, chest_mask)
    - For levels with monsters: (x, y, apple_mask, has_key, chest_mask, monster_positions_tuple)
    """
    
    def __init__(self, layout: List[str], monster_move_prob: float = 0.4):
        """
        Initialize GridWorld from layout.
        
        Args:
            layout: List of strings representing the grid
            monster_move_prob: Probability that each monster moves after agent action
        """
        self.layout = layout
        self.h = len(layout)
        self.w = len(layout[0]) if layout else 0
        self.monster_move_prob = monster_move_prob
        
        # Object collections
        self.rocks: Set[Tuple[int, int]] = set()
        self.fires: Set[Tuple[int, int]] = set()
        self.keys: Set[Tuple[int, int]] = set()
        self.chests: Set[Tuple[int, int]] = set()
        self.apples: List[Tuple[int, int]] = []
        self.apple_index: Dict[Tuple[int, int], int] = {}
        self.chest_index: Dict[Tuple[int, int], int] = {}
        self.initial_monsters: List[Tuple[int, int]] = []
        self.start: Tuple[int, int] = (0, 0)
        
        # Parse layout
        self._parse_layout()
        
        # State variables (initialized in reset)
        self.agent: Tuple[int, int] = (0, 0)
        self.apple_mask: int = 0
        self.chest_mask: int = 0
        self.collected_keys: int = 0
        self.opened_chests: Set[Tuple[int, int]] = set()
        self.monsters: List[Tuple[int, int]] = []
        self.alive: bool = True
        self.step_count: int = 0
        
    def _parse_layout(self):
        """Parse the level layout and populate object sets."""
        for y, row in enumerate(self.layout):
            for x, ch in enumerate(row):
                pos = (x, y)
                if ch == 'S':
                    self.start = pos
                elif ch == 'A':
                    self.apple_index[pos] = len(self.apples)
                    self.apples.append(pos)
                elif ch == 'K':
                    self.keys.add(pos)
                elif ch == 'C':
                    self.chest_index[pos] = len(self.chests)
                    self.chests.add(pos)
                elif ch == 'R':
                    self.rocks.add(pos)
                elif ch == 'F':
                    self.fires.add(pos)
                elif ch == 'M':
                    self.initial_monsters.append(pos)
    
    def reset(self) -> Tuple:
        """Reset environment to initial state."""
        self.agent = self.start
        self.collected_keys = 0
        self.opened_chests = set()
        self.alive = True
        self.step_count = 0
        
        # Initialize apple mask (bit per apple)
        self.apple_mask = 0
        for i in range(len(self.apples)):
            self.apple_mask |= (1 << i)
        
        # Initialize chest mask
        self.chest_mask = 0
        for i in range(len(self.chests)):
            self.chest_mask |= (1 << i)
        
        # Reset monsters to initial positions
        self.monsters = list(self.initial_monsters)
        
        return self.encode_state()
    
    def encode_state(self) -> Tuple:
        """
        Encode current state as a tuple.
        
        Returns tuple with:
        - Agent position (x, y)
        - Apple collection mask
        - Number of keys held
        - Chest opened mask
        - Monster positions (if monsters present)
        """
        if self.monsters:
            # Include monster positions in state for stochastic environment
            monster_tuple = tuple(sorted(self.monsters))
            return (self.agent[0], self.agent[1], self.apple_mask, 
                   self.collected_keys, self.chest_mask, monster_tuple)
        else:
            return (self.agent[0], self.agent[1], self.apple_mask, 
                   self.collected_keys, self.chest_mask)
    
    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= pos[0] < self.w and 0 <= pos[1] < self.h
    
    def is_blocked(self, pos: Tuple[int, int]) -> bool:
        """Check if position is blocked by rock."""
        return pos in self.rocks
    
    def has_monster_at(self, pos: Tuple[int, int]) -> bool:
        """Check if there's a monster at position."""
        return pos in self.monsters
    
    def try_move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Attempt to move from position in given direction.
        
        Returns new position (same as input if move blocked).
        """
        dx, dy = ACTIONS[action]
        new_pos = (pos[0] + dx, pos[1] + dy)
        
        if not self.in_bounds(new_pos):
            return pos
        if self.is_blocked(new_pos):
            return pos
        
        return new_pos
    
    def move_monsters(self):
        """Move each monster with probability monster_move_prob."""
        new_monsters = []
        
        for monster_pos in self.monsters:
            if random.random() < self.monster_move_prob:
                # Get valid moves for this monster
                valid_moves = []
                for action in ALL_ACTIONS:
                    new_pos = self.try_move(monster_pos, action)
                    # Monsters can't move into rocks or other monsters
                    if new_pos != monster_pos and new_pos not in new_monsters:
                        valid_moves.append(new_pos)
                
                # If no valid moves, stay in place
                if valid_moves:
                    new_monsters.append(random.choice(valid_moves))
                else:
                    new_monsters.append(monster_pos)
            else:
                # Didn't move this turn
                new_monsters.append(monster_pos)
        
        self.monsters = new_monsters
    
    def check_death(self) -> bool:
        """Check if agent died (fire or monster collision)."""
        if self.agent in self.fires:
            return True
        if self.has_monster_at(self.agent):
            return True
        return False
    
    def check_win_condition(self) -> bool:
        """Check if all collectibles are obtained."""
        # All apples collected and all chests opened
        return self.apple_mask == 0 and self.chest_mask == 0
    
    def step(self, action: int) -> StepResult:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (0=up, 1=right, 2=down, 3=left)
        
        Returns:
            StepResult with next state, reward, done flag, and info
        """
        if not self.alive:
            return StepResult(self.encode_state(), 0.0, True, {"event": "already_dead"})
        
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        
        # 1. Agent moves
        self.agent = self.try_move(self.agent, action)
        
        # 2. Check for death (fire or monster)
        if self.check_death():
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "death"})
        
        # 3. Collect apple if present
        if self.agent in self.apple_index:
            idx = self.apple_index[self.agent]
            if (self.apple_mask >> idx) & 1:  # Check if apple not yet collected
                self.apple_mask &= ~(1 << idx)  # Clear bit
                reward += 1.0
                info["collected"] = "apple"
        
        # 4. Collect key if present
        if self.agent in self.keys:
            self.collected_keys += 1
            self.keys.remove(self.agent)  # Remove from set so can't collect again
            info["collected"] = "key"
        
        # 5. Open chest if present and have key
        if self.agent in self.chest_index and self.agent not in self.opened_chests:
            if self.collected_keys > 0:
                idx = self.chest_index[self.agent]
                if (self.chest_mask >> idx) & 1:  # Check if chest not yet opened
                    self.chest_mask &= ~(1 << idx)
                    self.collected_keys -= 1
                    self.opened_chests.add(self.agent)
                    reward += 2.0
                    info["collected"] = "chest"
        
        # 6. Move monsters (after agent action)
        if self.monsters:
            self.move_monsters()
            
            # Check for death after monster movement
            if self.check_death():
                self.alive = False
                return StepResult(self.encode_state(), reward, True, {"event": "death_by_monster"})
        
        # 7. Check win condition
        if self.check_win_condition():
            done = True
            info["event"] = "win"
        
        return StepResult(self.encode_state(), reward, done, info)
    
    def get_valid_actions(self, pos: Optional[Tuple[int, int]] = None) -> List[int]:
        """Get list of valid actions from given position (or current agent position)."""
        if pos is None:
            pos = self.agent
        
        valid = []
        for action in ALL_ACTIONS:
            new_pos = self.try_move(pos, action)
            if new_pos != pos:  # Move was not blocked
                valid.append(action)
        
        # Always include at least one action (even if all blocked, can try to move)
        return valid if valid else [A_UP]
    
    def render_text(self) -> str:
        """Return text representation of current state (for debugging)."""
        lines = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                pos = (x, y)
                if pos == self.agent:
                    row.append('P')  # Player
                elif pos in self.monsters:
                    row.append('M')
                elif pos in self.fires:
                    row.append('F')
                elif pos in self.rocks:
                    row.append('R')
                elif pos in self.apple_index and (self.apple_mask >> self.apple_index[pos]) & 1:
                    row.append('A')
                elif pos in self.keys:
                    row.append('K')
                elif pos in self.chest_index and (self.chest_mask >> self.chest_index[pos]) & 1:
                    row.append('C')
                else:
                    row.append('.')
            lines.append(''.join(row))
        return '\n'.join(lines)
