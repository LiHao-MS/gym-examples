import gymnasium as gym
from gymnasium import spaces
# import pygame
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
                "dealer": spaces.Box(1, 22, shape=(1,), dtype=int),
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
        if isinstance(self._dealer_show, np.ndarray):
            self._dealer_show = self._dealer_show.item()
        if isinstance(self.player_state, np.ndarray):
            self.player_state = self.player_state.item()
        return {"player": self.player_state, "dealer": self._dealer_show, "ace": self._ace}

    def _update_dealer_state(self):
        while self.dealer_state < 17:
            new_card = self.deck[self._index]
            self._index += 1
            self.dealer_state += new_card.clip(
                min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
            )
            if new_card == 1:
                if self.dealer_state + 10 <= 21:
                    self.dealer_state += 10
                    self._dealer_ace = True 
            self._dealer_ace = self._dealer_ace or new_card == 1
        if self.dealer_state > 21 and self._dealer_ace:
            self.dealer_state -= 10
        return self.dealer_state

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
        # super().reset(seed=seed)
        super().reset(seed=None)

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
        _dealer_state = np.array([self.deck[2], self.deck[3]]).clip(
            min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"]
        )
        self._dealer_show = self.deck[2].clip(min=BlackjackEnv.metadata["min_point"], max=BlackjackEnv.metadata["max_point"])
        self.dealer_state = _dealer_state.sum()

        self._ace = np.any(_player_state == 1)
        self._dealer_ace = np.any(_dealer_state == 1)
        if self._dealer_ace:
            self.dealer_state += 10
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
            self._update_dealer_state()

        terminated = action == 1 or self.player_state.item() > 21

        reward = 0
        if terminated:
            player_fine_score =  BlackjackEnv.get_real_point(self.player_state, self._ace)
            dealer_fine_score =  BlackjackEnv.get_real_point(self.dealer_state, self._dealer_ace)
            self.player_state = player_fine_score
            if (player_fine_score > dealer_fine_score and player_fine_score <= 21) or (dealer_fine_score > 21 and player_fine_score <= 21):
                reward = 1
            elif player_fine_score == dealer_fine_score or (player_fine_score > 21 and dealer_fine_score > 21):
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
