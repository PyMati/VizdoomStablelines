import vizdoom as vzd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2
import numpy as np

HEALTH_INDEX = 6
ARMOR_INDEX = 7
KILLCOUNT_INDEX = 0
DAMAGE_COUNT_INDEX = 4


class VizDoomEnv(Env):
    def __init__(self, config_path: str, render=False, with_bots=False):
        super().__init__()

        self.game: vzd.DoomGame = vzd.DoomGame()
        self.game.load_config(config_path)

        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        self.seed = 0

        self.game.init()

        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 3), dtype=np.uint8
        )
        self.action_space = Discrete(self.game.get_available_buttons_size())

        self.episode_count = 0

        # Game variables
        self.killcount = 0
        self.hitcount = 0
        self.damagecount = 0
        self.health = 100
        self.armor = 0
        self.damagetaken = 0
        self.ammo = 50

    def reset_game_variables(self):
        self.killcount = 0
        self.hitcount = 0
        self.damagecount = 0
        self.health = 100
        self.armor = 0
        self.damagetaken = 0
        self.ammo = 50

    def step(self, action):
        # Specify action and take step
        actions = np.identity(self.game.get_available_buttons_size())
        killing_reward = self.game.make_action(actions[action], 4)

        reward = killing_reward
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.scale_down(state)

            variables = self.game.get_state().game_variables
            killcount, hitcount, damagecount, damagetaken, health, armor, ammo = (
                variables
            )

            killcount_reward = (killcount - self.killcount) * 150
            self.killcount = killcount

            hitcount_reward = (hitcount - self.hitcount) * 50
            self.hitcount = hitcount

            damage_reward = (damagecount - self.damagecount) * 5
            self.damagecount = damagecount

            damage_taken_reward = -(damagetaken - self.damagetaken) * 5
            self.damagetaken = damagetaken

            health_reward = (health - self.health) * 4
            self.health = health

            armor_reward = (armor - self.armor) * 4
            self.armor = armor

            reward += (
                killcount_reward
                + hitcount_reward
                + damage_reward
                + damage_taken_reward
                + health_reward
                + armor_reward
            )

        else:
            state = np.zeros(self.observation_space.shape)

        done = self.game.is_episode_finished()
        reward *= 0.01
        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        self.reset_game_variables()
        print(f"EPISODE: {self.episode_count}")
        super().reset()
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.scale_down(state), {}

    def scale_down(self, observation):
        resize = cv2.resize(
            np.moveaxis(observation, 0, -1), (160, 100), interpolation=cv2.INTER_CUBIC
        )
        state = np.reshape(resize, (100, 160, 3))
        return state

    def close(self):
        self.game.close()
