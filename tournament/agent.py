from typing import List, Tuple
import cv2
import numpy as np
from stable_baselines3 import PPO
from ActorCriticPolicy import CustomActorCriticPolicy
from CnnPolicy import CustomCNN


class Agent:
    def __init__(self, actions: List[List[int]]) -> None:
        self.actions = actions
        self.agent = PPO.load("best_model_80000.zip")

    def choose_action(self, game_state) -> Tuple[List[int], int]:
        if game_state.screen_buffer is not None:
            screen = game_state.screen_buffer
            resize = cv2.resize(
                np.moveaxis(screen, 0, -1), (160, 100), interpolation=cv2.INTER_CUBIC
            )
            state = np.reshape(resize, (100, 160, 3))
            pred, _ = self.agent.predict(state, deterministic=True)

        return self.actions[pred], pred
