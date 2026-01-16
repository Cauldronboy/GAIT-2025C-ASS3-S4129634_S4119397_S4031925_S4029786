from environment.arenaEnv import ArenaEnv

env = ArenaEnv(render_mode="human")
episode = 50

for ep in range(episode):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Replace with your action selection logic
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {ep + 1}: Total Reward: {total_reward}")