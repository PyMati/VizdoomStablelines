from DoomGymEnv import VizDoomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from TrainingCallback import callback, LOG_DIR
from CnnPolicy import CustomCNN
from ActorCriticPolicy import CustomActorCriticPolicy

if __name__ == "__main__":

    def make_env(config_path, rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param config_path: (str) Path to the config file
        :param rank: (int) Index of the subprocess
        :param seed: (int) Seed for the RNG
        """

        def _init():
            env = VizDoomEnv(config_path=config_path, render=False)
            return env

        return _init

    # Number of parallel environments
    n_envs = 2

    # Create the vectorized environment
    env = VizDoomEnv(config_path="deathmatch.cfg", render=True)
    # env = SubprocVecEnv([make_env("deathmatch.cfg", i) for i in range(n_envs)])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[256, 192, 192, 128],
            vf=[256, 192, 192, 128],
        ),
    )

    model = PPO(
        "CnnPolicy",
        # CustomActorCriticPolicy,
        env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=1.74e-5,
        n_steps=7000,
        policy_kwargs=policy_kwargs,
        clip_range=0.15,
        gae_lambda=0.9,
    )
    # model = PPO.load("train/train_basic/best_model_130000.zip", env)

    model.learn(total_timesteps=700_000, callback=callback, progress_bar=True)
