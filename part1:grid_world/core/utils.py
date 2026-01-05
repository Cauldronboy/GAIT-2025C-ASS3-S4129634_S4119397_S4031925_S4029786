"""Utility functions for training."""

import json
import os
import random
import numpy as np
from typing import List


def load_config(config_file: str = "config.json") -> dict:
    if not os.path.isabs(config_file):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(project_root, config_file)
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_file}")


def get_level_config(level_num: int, config_file: str = "config.json") -> dict:
    config = load_config(config_file)
    level_key = f"level{level_num}"
    
    if level_key not in config:
        raise ValueError(f"Configuration for {level_key} not found")
    
    return config[level_key]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class TrainingLogger:
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[bool] = []
    
    def log_episode(self, reward: float, length: int, success: bool):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(success)
    
    def get_stats(self, last_n: int = None) -> dict:
        if last_n is not None:
            rewards = self.episode_rewards[-last_n:]
            lengths = self.episode_lengths[-last_n:]
            successes = self.episode_successes[-last_n:]
        else:
            rewards = self.episode_rewards
            lengths = self.episode_lengths
            successes = self.episode_successes
        
        if not rewards:
            return {}
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes) if successes else 0.0,
            "total_episodes": len(rewards)
        }
    
    def print_progress(self, episode: int, total_episodes: int, window: int = 100):
        if episode % window == 0 and episode > 0:
            stats = self.get_stats(last_n=window)
            print(f"Episode {episode}/{total_episodes} | "
                  f"Reward: {stats['mean_reward']:.2f}±{stats['std_reward']:.2f} | "
                  f"Length: {stats['mean_length']:.1f} | "
                  f"Success: {stats['success_rate']*100:.1f}%")
