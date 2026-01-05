"""SARSA algorithm implementation."""

import random
from typing import Dict, Tuple, List
from environment.gridworld import ALL_ACTIONS


class SARSATable:
    def __init__(self):
        self.q: Dict[Tuple[Tuple, int], float] = {}
    
    def get(self, state: Tuple, action: int) -> float:
        return self.q.get((state, action), 0.0)
    
    def set(self, state: Tuple, action: int, value: float):
        self.q[(state, action)] = value
    
    def get_best_value(self, state: Tuple) -> float:
        return max(self.get(state, a) for a in ALL_ACTIONS)
    
    def get_best_actions(self, state: Tuple) -> List[int]:
        q_values = [self.get(state, a) for a in ALL_ACTIONS]
        max_q = max(q_values)
        return [a for a, q in zip(ALL_ACTIONS, q_values) if q == max_q]
    
    def size(self) -> int:
        return len(self.q)


class SARSAAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 1000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        self.qtable = SARSATable()
        self.current_episode = 0
    
    def get_epsilon(self, episode: int = None) -> float:
        if episode is None:
            episode = self.current_episode
        
        if self.epsilon_decay_episodes <= 0:
            return self.epsilon_end
        
        progress = min(episode / self.epsilon_decay_episodes, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, state: Tuple, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return random.choice(ALL_ACTIONS)
        
        best_actions = self.qtable.get_best_actions(state)
        return random.choice(best_actions)
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, next_action: int, done: bool):
        # SARSA: on-policy, uses actual next action Q(s',a') for target
        current_q = self.qtable.get(state, action)
        
        if done:
            target = reward
        else:
            next_q = self.qtable.get(next_state, next_action)
            target = reward + self.gamma * next_q
        
        new_q = current_q + self.alpha * (target - current_q)
        self.qtable.set(state, action, new_q)
    
    def get_greedy_action(self, state: Tuple) -> int:
        best_actions = self.qtable.get_best_actions(state)
        return random.choice(best_actions)
    
    def reset_episode(self):
        self.current_episode += 1
    
    def save_qtable(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.qtable.q, f)
    
    def load_qtable(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            self.qtable.q = pickle.load(f)
