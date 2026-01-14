import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("PilotEnv-v0")  # Change to actual enviroment id
env.reset()

models_dir = f"models/PPO"
models_path = f"{models_dir}" # Change to desired model path

model = PPO.load(models_path, env=env)

episode = 10

for ep in range(episode):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

env.close()