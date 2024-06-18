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
            "render_modes": ["human", "rgb_array"], 
            "render_fps": 4
        }

    def __init__(self, render_mode=None, size=5, seed=72):

        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(1, 22, shape=(1,), dtype=int),
                "banker": spaces.Box(1, 22, shape=(1,), dtype=int),
                "ace": spaces.Discrete(2),
            }
        )

        # We have 2 actions, corresponding to "twist", "stick"
        self.action_space = spaces.Discrete(2)
        self._index = 0
        self.seed = seed
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "twist", 1 to "stick" etc.
        """
        self._action_to_direction = {
            0: 0,
            1: 1,
        }
        self.render_mode = render_mode

    def _get_obs(self):
        if isinstance(self._ace, np.ndarray):
            self._ace = self._ace.item()
        if isinstance(self._banker_show, np.ndarray):
            self._banker_show = self._banker_show.item()
        if isinstance(self.player_state, np.ndarray):
            self.player_state = self.player_state.item()
        return {"player": self.player_state, "banker": self._banker_show, "ace": self._ace}

    def _update_banker_state(self):
        while self.banker_state < 17:
            new_card = self.deck[self._index]
            self._index += 1
            self.banker_state += new_card.clip(
                min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
            )
            if new_card == 1:
                if self.banker_state + 10 <= 21:
                    self.banker_state += 10
                    self._banker_ace = True 
            self._banker_ace = self._banker_ace or new_card == 1
        if self.banker_state > 21 and self._banker_ace:
            self.banker_state -= 10
        return self.banker_state

    @staticmethod
    def get_real_point(obs: int, ace: bool):
        if not ace or (21 - obs < 10 and obs <= 21):
            return obs
        elif obs > 21:
            return 22
        else:
            return obs + 10

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        np.random.seed(self.seed)
        self.seed = np.random.randint(0, 2**32 - 1)
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
        self.player_state = _player_state.sum()
        _banker_state = np.array([self.deck[2], self.deck[3]]).clip(
            min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
        )
        self._banker_show = self.deck[2].clip(min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"])
        self.banker_state = _banker_state.sum()

        self._ace = np.any(_player_state == 1)
        self._banker_ace = np.any(_banker_state == 1)
        if self._banker_ace:
            self.banker_state += 10
        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        if action == 0:
            new_card = self.deck[self._index]
            self._index += 1
            self.player_state += new_card.clip(
                min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
            )

            self._ace = self._ace or new_card == 1 
        else:
            self._update_banker_state()

        terminated = action == 1 or self.player_state.item() > 21

        reward = 0
        if terminated:
            player_fine_score =  BlackjackEnv.get_real_point(self.player_state, self._ace)
            banker_fine_score =  BlackjackEnv.get_real_point(self.banker_state, self._banker_ace)
            self.player_state = player_fine_score
            if player_fine_score > banker_fine_score and player_fine_score <= 21 and banker_fine_score <= 21:
                reward = 1
            elif player_fine_score == banker_fine_score or (player_fine_score > 21 and banker_fine_score > 21):
                reward = 0
            else:
                reward = -1

        observation = self._get_obs()
        info = {}

        if isinstance(terminated, np.ndarray):
            terminated = terminated.item()
        return observation, reward, terminated, False, info

    def close(self):
        return None
