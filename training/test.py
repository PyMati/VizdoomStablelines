from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from DoomGymEnv import VizDoomEnv

env = VizDoomEnv(config_path="deadly_corridor.cfg", render=True)
model = PPO.load("train/train_basic/best_model_130000.zip")

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward}")
