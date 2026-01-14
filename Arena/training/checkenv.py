from stable_baselines3.common.env_checker import check_env
from Arena.environment.arena import Arena

env = Arena()
check_env(env)
