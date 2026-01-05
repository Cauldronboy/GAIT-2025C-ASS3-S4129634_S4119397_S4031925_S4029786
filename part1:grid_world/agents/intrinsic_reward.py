"""
Intrinsic reward mechanism for exploration.

Implements count-based exploration bonus:
    intrinsic_reward = 1 / sqrt(n(s))

Where n(s) is the number of times state s has been visited in current episode.
"""

import math
from typing import Dict, Tuple
from collections import defaultdict


class IntrinsicRewardTracker:
    """Tracks state visits and computes intrinsic rewards."""
    
    def __init__(self):
        """Initialize visit counters."""
        # Visit counts for current episode
        self.episode_visits: Dict[Tuple, int] = defaultdict(int)
        
        # Total visits across all episodes (for analysis)
        self.total_visits: Dict[Tuple, int] = defaultdict(int)
        
    def reset_episode(self):
        """Reset episode visit counter (call at start of each episode)."""
        self.episode_visits.clear()
    
    def visit_state(self, state: Tuple):
        """
        Record a visit to state.
        
        Args:
            state: State tuple
        """
        self.episode_visits[state] += 1
        self.total_visits[state] += 1
    
    def get_intrinsic_reward(self, state: Tuple) -> float:
        # Intrinsic reward: 1 / sqrt(n) where n = visit count in current episode
        n = self.episode_visits.get(state, 0)
        if n == 0:
            return 1.0
        else:
            return 1.0 / math.sqrt(n)
    
    def get_combined_reward(self, state: Tuple, env_reward: float) -> float:
        """
        Get combined environment + intrinsic reward.
        
        Args:
            state: State tuple
            env_reward: Reward from environment
        
        Returns:
            Total reward (environment + intrinsic)
        """
        intrinsic = self.get_intrinsic_reward(state)
        return env_reward + intrinsic
    
    def get_visit_count(self, state: Tuple, total: bool = False) -> int:
        """
        Get number of visits to state.
        
        Args:
            state: State tuple
            total: If True, return total visits across all episodes
                   If False, return visits in current episode
        
        Returns:
            Visit count
        """
        if total:
            return self.total_visits.get(state, 0)
        else:
            return self.episode_visits.get(state, 0)
    
    def get_exploration_coverage(self) -> Dict[str, float]:
        """
        Get exploration statistics.
        
        Returns:
            Dictionary with exploration metrics
        """
        unique_states = len(self.total_visits)
        total_visits = sum(self.total_visits.values())
        
        if unique_states == 0:
            return {
                "unique_states": 0,
                "total_visits": 0,
                "avg_visits_per_state": 0.0
            }
        
        return {
            "unique_states": unique_states,
            "total_visits": total_visits,
            "avg_visits_per_state": total_visits / unique_states
        }


class IntrinsicQLearningAgent:
    """Q-Learning agent with intrinsic reward."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 1000):
        """
        Initialize Q-Learning agent with intrinsic rewards.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_episodes: Episodes over which to decay epsilon
        """
        # Import here to avoid circular dependency
        from .q_learning import QLearningAgent
        
        # Use standard Q-learning agent
        self.agent = QLearningAgent(alpha, gamma, epsilon_start, 
                                    epsilon_end, epsilon_decay_episodes)
        
        # Add intrinsic reward tracker
        self.intrinsic_tracker = IntrinsicRewardTracker()
    
    def reset_episode(self):
        """Reset for new episode."""
        self.agent.reset_episode()
        self.intrinsic_tracker.reset_episode()
    
    def get_epsilon(self, episode: int = None) -> float:
        """Get current epsilon."""
        return self.agent.get_epsilon(episode)
    
    def select_action(self, state: Tuple, epsilon: float = None) -> int:
        """Select action using epsilon-greedy."""
        return self.agent.select_action(state, epsilon)
    
    def update(self, state: Tuple, action: int, env_reward: float, 
               next_state: Tuple, done: bool):
        """
        Update with intrinsic reward.
        
        Args:
            state: Current state
            action: Action taken
            env_reward: Environment reward
            next_state: Next state
            done: Whether episode ended
        """
        # Record visit to next state
        self.intrinsic_tracker.visit_state(next_state)
        
        # Calculate intrinsic reward
        intrinsic_reward = self.intrinsic_tracker.get_intrinsic_reward(next_state)
        
        # Combined reward
        total_reward = env_reward + intrinsic_reward
        
        # Standard Q-learning update with combined reward
        self.agent.update(state, action, total_reward, next_state, done)
    
    def get_greedy_action(self, state: Tuple) -> int:
        """Get greedy action."""
        return self.agent.get_greedy_action(state)
    
    @property
    def qtable(self):
        """Access to Q-table."""
        return self.agent.qtable


class IntrinsicSARSAAgent:
    """SARSA agent with intrinsic reward."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 1000):
        """
        Initialize SARSA agent with intrinsic rewards.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_episodes: Episodes over which to decay epsilon
        """
        # Import here to avoid circular dependency
        from .sarsa import SARSAAgent
        
        # Use standard SARSA agent
        self.agent = SARSAAgent(alpha, gamma, epsilon_start, 
                               epsilon_end, epsilon_decay_episodes)
        
        # Add intrinsic reward tracker
        self.intrinsic_tracker = IntrinsicRewardTracker()
    
    def reset_episode(self):
        """Reset for new episode."""
        self.agent.reset_episode()
        self.intrinsic_tracker.reset_episode()
    
    def get_epsilon(self, episode: int = None) -> float:
        """Get current epsilon."""
        return self.agent.get_epsilon(episode)
    
    def select_action(self, state: Tuple, epsilon: float = None) -> int:
        """Select action using epsilon-greedy."""
        return self.agent.select_action(state, epsilon)
    
    def update(self, state: Tuple, action: int, env_reward: float, 
               next_state: Tuple, next_action: int, done: bool):
        """
        Update with intrinsic reward.
        
        Args:
            state: Current state
            action: Action taken
            env_reward: Environment reward
            next_state: Next state
            next_action: Next action
            done: Whether episode ended
        """
        # Record visit to next state
        self.intrinsic_tracker.visit_state(next_state)
        
        # Calculate intrinsic reward
        intrinsic_reward = self.intrinsic_tracker.get_intrinsic_reward(next_state)
        
        # Combined reward
        total_reward = env_reward + intrinsic_reward
        
        # Standard SARSA update with combined reward
        self.agent.update(state, action, total_reward, next_state, next_action, done)
    
    def get_greedy_action(self, state: Tuple) -> int:
        """Get greedy action."""
        return self.agent.get_greedy_action(state)
    
    @property
    def qtable(self):
        """Access to Q-table."""
        return self.agent.qtable
