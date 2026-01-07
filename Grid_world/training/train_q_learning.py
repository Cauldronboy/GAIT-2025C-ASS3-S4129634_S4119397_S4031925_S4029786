"""Q-Learning training script for all levels."""

import argparse
import os
import pygame
from typing import List

from environment import GridWorld, get_level, get_level_name, LEVELS
from agents import QLearningAgent, IntrinsicQLearningAgent
from environment import GridWorldRenderer
from core import get_level_config, set_seed, TrainingLogger


def train_q_learning(level_num: int, visualize: bool = True, use_intrinsic: bool = False):
    config = get_level_config(level_num)
    level_name = get_level_name(level_num)
    
    set_seed(config['seed'])
    
    layout = get_level(level_num)
    monster_prob = config.get('monsterMoveProb', 0.4)
    env = GridWorld(layout, monster_move_prob=monster_prob)
    
    if use_intrinsic or config.get('useIntrinsicReward', False):
        agent = IntrinsicQLearningAgent(
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon_start=config['epsilonStart'],
            epsilon_end=config['epsilonEnd'],
            epsilon_decay_episodes=config['epsilonDecayEpisodes']
        )
        algorithm_name = "Q-Learning + Intrinsic"
    else:
        agent = QLearningAgent(
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon_start=config['epsilonStart'],
            epsilon_end=config['epsilonEnd'],
            epsilon_decay_episodes=config['epsilonDecayEpisodes']
        )
        algorithm_name = "Q-Learning"
    
    renderer = None
    if visualize:
        renderer = GridWorldRenderer(tile_size=config['tileSize'])
        renderer.init_display(env, title=f"{algorithm_name} - {level_name}")
    
    episodes = config['episodes']
    max_steps = config['maxStepsPerEpisode']
    fps_visual = config['fpsVisual']
    fps_fast = config['fpsFast']
    
    logger = TrainingLogger()
    running = True
    show_visual = visualize
    
    print(f"\nTraining {algorithm_name} on {level_name}")
    print(f"Episodes: {episodes}, Alpha: {config['alpha']}, Gamma: {config['gamma']}")
    print(f"Epsilon: {config['epsilonStart']} -> {config['epsilonEnd']}")
    print("Press V to toggle fast mode, R to reset, ESC to quit\n")
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        agent.reset_episode()
        
        episode_reward = 0.0
        step = 0
        epsilon = agent.get_epsilon(episode)
        
        while running and step < max_steps:
            # Handle events
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        if event.key == pygame.K_v:
                            show_visual = not show_visual
                        if event.key == pygame.K_r:
                            # Reset training
                            state = env.reset()
                            agent.reset_episode()
                            episode_reward = 0.0
                            step = 0
            
            if not running:
                break
            
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take step
            result = env.step(action)
            
            # Update agent
            agent.update(state, action, result.reward, result.next_state, result.done)
            
            # Track reward
            episode_reward += result.reward
            state = result.next_state
            step += 1
            
            # Render
            if visualize:
                if show_visual or step % 10 == 0:
                    extra_info = f"Q-table size: {agent.qtable.size()}"
                    renderer.render(env, episode, episodes, step, epsilon, 
                                  episode_reward, algorithm_name, level_name, extra_info)
                    renderer.tick(fps_visual if show_visual else fps_fast)
            
            # Check if done
            if result.done:
                break
        
        # Log episode
        success = env.check_win_condition()
        logger.log_episode(episode_reward, step, success)
        
        # Print progress
        if episode % 100 == 0 or episode == episodes - 1:
            logger.print_progress(episode, episodes)
        
        if not running:
            break
    
    # Cleanup
    if renderer:
        renderer.close()
    
    # Print final statistics
    stats = logger.get_stats()
    print(f"\n{'='*60}")
    print(f"Training Complete: {algorithm_name} on {level_name}")
    print(f"{'='*60}")
    print(f"Total Episodes: {len(logger.episode_rewards)}")
    print(f"Mean Reward: {stats['mean_reward']:.2f}")
    print(f"Success Rate: {stats['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Q-Learning agent')
    parser.add_argument('--level', type=int, default=None, 
                       help='Level to train (0-6). If not specified, trains all levels.')
    parser.add_argument('--no-visual', action='store_true',
                       help='Disable visualization for faster training')
    parser.add_argument('--intrinsic', action='store_true',
                       help='Use intrinsic rewards')
    
    args = parser.parse_args()
    
    if args.level is not None:
        # Train single level
        if args.level not in LEVELS:
            print(f"Error: Invalid level {args.level}. Available levels: {list(LEVELS.keys())}")
            return
        
        train_q_learning(args.level, visualize=not args.no_visual, 
                        use_intrinsic=args.intrinsic)
    else:
        # Train all levels
        for level_num in sorted(LEVELS.keys()):
            # Level 6 uses intrinsic rewards by default
            use_intrinsic = args.intrinsic or (level_num == 6)
            train_q_learning(level_num, visualize=not args.no_visual,
                           use_intrinsic=use_intrinsic)
            print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
