import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class BlackjackEnv(gym.Env):
    metadata = {
            "min_card": 1,
            "max_card": 13,
            "min_point": 1,
            "max_point": 10,
            "target_point": 21,
        }

    def __init__(self):
       
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(1, 10, shape=(1,), dtype=int),
                "banker": spaces.Box(1, 10, shape=(1,), dtype=int),
            }
        )

        # We have 2 actions, corresponding to "twist", "stick"
        self.action_space = spaces.Discrete(2)
        self._index = 0
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "twist", 1 to "stick" etc.
        """
        self._action_to_direction = {
            0: np.array([0]),
            1: np.array([1]),
        }

    def _get_obs(self):
        return {"player": self._player_state, "banker": self._banker_show, "ace": self._ace}

    def _upodate_banker_state(self):
        if self._banker_state < 17:
            new_card = self.deck[self._index]
            self._index += 1
            self._banker_state += new_card.clip(
                min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
            )
            self._banker_ace = self._banker_ace or new_card == 1
        return self._banker_state

    @staticmethod
    def get_real_point(obs: int, ace: bool):
        if not ace or np.abs(obs - 21) < 10:
            return obs
        else:
            return obs + 10

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Generate the deck: 1-13 repeated four times
        self.deck = np.tile(
            np.arange(self.metadata["min_card"], self.metadata["max_card"] + 1), 4
        )

        # Shuffle the deck
        self.np_random.shuffle(self.deck)

        self._index = 4
        _player_state = np.array([self.deck[0], self.deck[1]]).clip(
            min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
        )
        self._player_state = _player_state.sum()
        _banker_state = np.array([self.deck[2], self.deck[3]]).clip(
            min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
        )
        self._banker_show = self.deck[2]
        self._banker_state = _banker_state.sum()

        self._ace = np.any(self._player_state == 1)
        self._banker_ace = np.any(self._banker_state == 1)
        observation = self._get_obs()
        info = None

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        if action == 0:
            new_card = self.deck[self._index]
            self._index += 1
            self._player_state += new_card.clip(
                min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
            )
            self._ace = self._ace or new_card == 1 
        else:
            self._upodate_banker_state()

        terminated = (
            action != 0 
            or self._player_state > 21
        )

        reward = 0
        if terminated:
            player_fine_score =  BlackjackEnv.get_real_point(self._player_state, self.ace)
            banker_fine_score =  BlackjackEnv.get_real_point(self._banker_state, self._banker_ace)
            if player_fine_score > banker_fine_score:
                reward = 1
            elif player_fine_score == banker_fine_score:
                reward = 0
            else:
                reward = -1
                
        observation = self._get_obs()
        info = None

        return observation, reward, terminated, False, info

    def close(self):
        return None
