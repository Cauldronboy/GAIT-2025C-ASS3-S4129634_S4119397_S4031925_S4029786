from stable_baselines3.common.env_checker import check_env
from environment.arenaEnv import ArenaEnv

env = ArenaEnv()
check_env(env)
