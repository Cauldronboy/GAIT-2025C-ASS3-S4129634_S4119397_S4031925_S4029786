"""
Q-Learning algorithm implementation.

Features:
- Epsilon-greedy exploration with linear decay
- Random tie-breaking for equal Q-values
- Standard Q-learning update rule
"""

import random
from typing import Dict, Tuple, List
from environment.gridworld import ALL_ACTIONS


class QTable:
    """Q-table for storing state-action values."""
    
    def __init__(self):
        """Initialize empty Q-table."""
        self.q: Dict[Tuple[Tuple, int], float] = {}
    
    def get(self, state: Tuple, action: int) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: State tuple
            action: Action index
        
        Returns:
            Q-value (0.0 if not seen before)
        """
        return self.q.get((state, action), 0.0)
    
    def set(self, state: Tuple, action: int, value: float):
        """
        Set Q-value for state-action pair.
        
        Args:
            state: State tuple
            action: Action index
            value: Q-value to set
        """
        self.q[(state, action)] = value
    
    def get_best_value(self, state: Tuple) -> float:
        """
        Get maximum Q-value for state across all actions.
        
        Args:
            state: State tuple
        
        Returns:
            Maximum Q-value
        """
        return max(self.get(state, a) for a in ALL_ACTIONS)
    
    def get_best_actions(self, state: Tuple) -> List[int]:
        """
        Get all actions with maximum Q-value (for tie-breaking).
        
        Args:
            state: State tuple
        
        Returns:
            List of actions with equal maximum Q-value
        """
        q_values = [self.get(state, a) for a in ALL_ACTIONS]
        max_q = max(q_values)
        return [a for a, q in zip(ALL_ACTIONS, q_values) if q == max_q]
    
    def size(self) -> int:
        """Return number of state-action pairs in table."""
        return len(self.q)


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy exploration."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 1000):
        """
        Initialize Q-Learning agent.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_episodes: Episodes over which to decay epsilon
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        self.qtable = QTable()
        self.current_episode = 0
    
    def get_epsilon(self, episode: int = None) -> float:
        """
        Get current epsilon value with linear decay.
        
        Args:
            episode: Episode number (uses current_episode if None)
        
        Returns:
            Current epsilon value
        """
        if episode is None:
            episode = self.current_episode
        
        if self.epsilon_decay_episodes <= 0:
            return self.epsilon_end
        
        # Linear decay
        progress = min(episode / self.epsilon_decay_episodes, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, state: Tuple, epsilon: float = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (uses current epsilon if None)
        
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        # Exploration: random action
        if random.random() < epsilon:
            return random.choice(ALL_ACTIONS)
        
        # Exploitation: best action with random tie-breaking
        best_actions = self.qtable.get_best_actions(state)
        return random.choice(best_actions)
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, done: bool):
        """
        Update Q-value using Q-learning rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        current_q = self.qtable.get(state, action)
        
        if done:
            # Terminal state: no future reward
            target = reward
        else:
            # Use max over next state's Q-values
            max_next_q = self.qtable.get_best_value(next_state)
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        new_q = current_q + self.alpha * (target - current_q)
        self.qtable.set(state, action, new_q)
    
    def get_greedy_action(self, state: Tuple) -> int:
        """
        Get greedy action (no exploration).
        
        Args:
            state: Current state
        
        Returns:
            Best action according to Q-table
        """
        best_actions = self.qtable.get_best_actions(state)
        return random.choice(best_actions)
    
    def reset_episode(self):
        """Increment episode counter (for epsilon decay)."""
        self.current_episode += 1
    
    def save_qtable(self, filepath: str):
        """Save Q-table to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.qtable.q, f)
    
    def load_qtable(self, filepath: str):
        """Load Q-table from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.qtable.q = pickle.load(f)


def linear_epsilon_decay(episode: int, start: float, end: float, 
                         decay_episodes: int) -> float:
    """
    Calculate epsilon with linear decay.
    
    Args:
        episode: Current episode
        start: Starting epsilon
        end: Ending epsilon
        decay_episodes: Episodes to decay over
    
    Returns:
        Current epsilon value
    """
    if decay_episodes <= 0:
        return end
    
    progress = min(episode / decay_episodes, 1.0)
    return start + progress * (end - start)
