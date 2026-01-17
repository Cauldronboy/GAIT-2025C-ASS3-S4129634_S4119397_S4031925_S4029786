from stable_baselines3.common.env_checker import check_env
from Arena.environment.arena import ArenaEnv

env = ArenaEnv(render_mode="human")
check_env(env)
