from stable_baselines3.common.env_checker import check_env
from environment import ArenaEnv

env = ArenaEnv(render_mode="human")
check_env(env)
