#!/usr/bin/env python3
"""
GAIT Assignment 3 - Part 1: Classical Reinforcement Learning
Main entry point for running Q-Learning and SARSA on gridworld levels.

Usage:
    python main.py              # GUI menu to select task
    python main.py 0            # Run Task 1: Q-Learning Level 0
    python main.py 1            # Run Task 2: SARSA Level 1
    python main.py 2            # Run Task 3: Q-Learning Level 2
    python main.py 3            # Run Task 3: Q-Learning Level 3
    python main.py 4            # Run Task 4: Q-Learning Level 4
    python main.py 5            # Run Task 4: SARSA Level 5
    python main.py 6            # Run Task 5: Q-Learning Level 6 (intrinsic)
"""

import sys
import pygame
from typing import Optional, Tuple
from environment import GridWorld, get_level, get_level_name, GridWorldRenderer
from agents import QLearningAgent, SARSAAgent, IntrinsicQLearningAgent
from core import get_level_config, set_seed, TrainingLogger


class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, task_num: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.task_num = task_num
        self.hovered = False
        
        self.color_normal = (45, 50, 58)
        self.color_hover = (74, 222, 128)
        self.color_text = (240, 240, 240)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        color = self.color_hover if self.hovered else self.color_normal
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2, border_radius=8)
        
        text_surface = font.render(self.text, True, self.color_text)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def update_hover(self, mouse_pos: Tuple[int, int]):
        self.hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked(self, mouse_pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


def draw_menu(screen: pygame.Surface, buttons: list, font: pygame.font.Font, 
              font_title: pygame.font.Font):
    screen.fill((25, 28, 34))
    
    title = font_title.render("GAIT Assignment 3 - Reinforcement Learning", True, (74, 222, 128))
    title_rect = title.get_rect(center=(400, 50))
    screen.blit(title, title_rect)
    
    subtitle = font.render("Select a task to run:", True, (200, 200, 200))
    subtitle_rect = subtitle.get_rect(center=(400, 100))
    screen.blit(subtitle, subtitle_rect)
    
    for button in buttons:
        button.draw(screen, font)
    
    info = font.render("ESC - Quit | Click a task to begin", True, (156, 163, 175))
    info_rect = info.get_rect(center=(400, 560))
    screen.blit(info, info_rect)
    
    pygame.display.flip()


def show_menu() -> Optional[int]:
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("GridWorld RL - Main Menu")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("arial", 18)
    font_title = pygame.font.SysFont("arial", 28, bold=True)
    
    button_width, button_height = 600, 50
    start_y = 140
    spacing = 60
    
    task_descriptions = [
        "Task 1: Q-Learning on Level 0 (basic apples)",
        "Task 2: SARSA on Level 1 (fires & hazards)",
        "Task 3: Q-Learning on Level 2 (keys & chests)",
        "Task 3: Q-Learning on Level 3 (complex puzzle)",
        "Task 4: Q-Learning on Level 4 (monsters)",
        "Task 4: SARSA on Level 5 (monsters)",
        "Task 5: Q-Learning on Level 6 (intrinsic rewards)"
    ]
    
    buttons = []
    for i, desc in enumerate(task_descriptions):
        x = (800 - button_width) // 2
        y = start_y + i * spacing
        buttons.append(Button(x, y, button_width, button_height, desc, i))
    
    running = True
    selected_task = None
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                selected_task = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    selected_task = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    for button in buttons:
                        if button.is_clicked(mouse_pos):
                            selected_task = button.task_num
                            running = False
                            break
        
        # Update button hover states
        for button in buttons:
            button.update_hover(mouse_pos)
        
        # Draw
        draw_menu(screen, buttons, font, font_title)
        clock.tick(60)
    
    pygame.quit()
    return selected_task


def run_task(level_num: int, use_sarsa: bool = False, use_intrinsic: bool = False):
    config = get_level_config(level_num)
    level_name = get_level_name(level_num)
    set_seed(config['seed'])
    
    layout = get_level(level_num)
    monster_prob = config.get('monsterMoveProb', 0.4)
    env = GridWorld(layout, monster_move_prob=monster_prob)
    
    if use_intrinsic:
        agent = IntrinsicQLearningAgent(
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon_start=config['epsilonStart'],
            epsilon_end=config['epsilonEnd'],
            epsilon_decay_episodes=config['epsilonDecayEpisodes']
        )
        algorithm_name = "Q-Learning + Intrinsic"
    elif use_sarsa:
        agent = SARSAAgent(
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon_start=config['epsilonStart'],
            epsilon_end=config['epsilonEnd'],
            epsilon_decay_episodes=config['epsilonDecayEpisodes']
        )
        algorithm_name = "SARSA"
    else:
        agent = QLearningAgent(
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon_start=config['epsilonStart'],
            epsilon_end=config['epsilonEnd'],
            epsilon_decay_episodes=config['epsilonDecayEpisodes']
        )
        algorithm_name = "Q-Learning"
    
    renderer = GridWorldRenderer(tile_size=config['tileSize'])
    renderer.init_display(env, title=f"{algorithm_name} - {level_name}")
    episodes = config['episodes']
    max_steps = config['maxStepsPerEpisode']
    fps_visual = config['fpsVisual']
    fps_fast = config['fpsFast']
    
    logger = TrainingLogger()
    running = True
    show_visual = True
    
    print(f"\nRunning {algorithm_name} on {level_name}")
    print(f"Episodes: {episodes}, Alpha: {config['alpha']}, Gamma: {config['gamma']}")
    print(f"Controls: V=toggle speed | R=reset | ESC=quit\n")
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset_episode()
        
        episode_reward = 0.0
        step = 0
        epsilon = agent.get_epsilon(episode)
        
        # SARSA needs initial action before loop
        if use_sarsa:
            action = agent.select_action(state, epsilon)
        
        while running and step < max_steps:
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
                        state = env.reset()
                        agent.reset_episode()
                        episode_reward = 0.0
                        step = 0
                        if use_sarsa:
                            action = agent.select_action(state, epsilon)
            
            if not running:
                break
            
            # Q-learning selects action here; SARSA uses pre-selected action
            if not use_sarsa:
                action = agent.select_action(state, epsilon)
            
            result = env.step(action)
            
            if use_sarsa:
                # SARSA updates with actual next action (on-policy)
                next_action = agent.select_action(result.next_state, epsilon)
                agent.update(state, action, result.reward, result.next_state, 
                           next_action, result.done)
                action = next_action
            else:
                # Q-learning updates with max Q-value (off-policy)
                agent.update(state, action, result.reward, result.next_state, result.done)
            
            episode_reward += result.reward
            state = result.next_state
            step += 1
            
            if show_visual or step % 10 == 0:
                extra_info = f"Q-table size: {agent.qtable.size()}"
                renderer.render(env, episode, episodes, step, epsilon, 
                              episode_reward, algorithm_name, level_name, extra_info)
                renderer.tick(fps_visual if show_visual else fps_fast)
            
            if result.done:
                break
        
        success = env.check_win_condition()
        logger.log_episode(episode_reward, step, success)
        
        if episode % 100 == 0 or episode == episodes - 1:
            logger.print_progress(episode, episodes)
        
        if not running:
            break
    
    renderer.close()
    
    stats = logger.get_stats()
    print(f"\n{'='*60}")
    print(f"Training Complete: {algorithm_name} on {level_name}")
    print(f"{'='*60}")
    print(f"Total Episodes: {len(logger.episode_rewards)}")
    print(f"Mean Reward: {stats['mean_reward']:.2f}")
    print(f"Success Rate: {stats['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    tasks = {
        0: (0, False, False),
        1: (1, True, False),
        2: (2, False, False),
        3: (3, False, False),
        4: (4, False, False),
        5: (5, True, False),
        6: (6, False, True),
    }
    
    if len(sys.argv) > 1:
        try:
            task_num = int(sys.argv[1])
            if task_num not in tasks:
                print(f"Error: Invalid task number {task_num}")
                print("Valid tasks: 0-6")
                sys.exit(1)
            
            level, use_sarsa, use_intrinsic = tasks[task_num]
            run_task(level, use_sarsa, use_intrinsic)
        except ValueError:
            print("Error: Task number must be an integer (0-6)")
            sys.exit(1)
    else:
        while True:
            selected_task = show_menu()
            
            if selected_task is None:
                break
            
            level, use_sarsa, use_intrinsic = tasks[selected_task]
            run_task(level, use_sarsa, use_intrinsic)

if __name__ == "__main__":
    main()
