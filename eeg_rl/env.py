import numpy as np
from myosuite.utils import gym
from gymnasium.spaces import Box
from .data import (
    select_random_eeg_mat_file, load_random_session_data, distance_matrix as dm,
    simplex_adjacency_history as sah, sinusoidal_positional_encoding
)
from typing import Optional


class EEGTdaEnv(gym.Env):
    """
    EEG-based Reinforcement Learning Environment using MyoSuite.
    """
    __step__: int = 0
    eeg_data_windows: Optional[np.ndarray] = None
    metadata = {"render_modes": []}

    def __init__(self, window_length: int = 10, n_filters: int = 64, eeg_channels: int = 64):
        """
        Initialize the EEG RL Environment.
        :param window_length: EEG signal window time length in ms
        :param n_filters: Number of filters used for TDA
        """
        super().__init__()
        self.env = gym.make("myoLegWalk-v0")
        self.action_space = self.env.action_space
        self.observation_space = Box(
            low=-1.0, high=1.0, 
            shape=(n_filters, window_length, window_length), dtype=np.float32)
        self.window_length = window_length
        self.n_filters = n_filters
        self.eeg_channels = eeg_channels

    def __get_obs__(self):
        """
        Get the current observation from the EEG data windows.
        :return: Current EEG observation
        """
        if self.__step__ is None:
            raise ValueError("Environment not reset. Call reset() before getting observations.")
        eeg_obs = self.eeg_data_windows[self.__step__]
        eeg_adj_history = sah(dm(eeg_obs), self.n_filters)
        return eeg_adj_history

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        :param seed: Random seed for reproducibility
        :param options: Additional options for resetting the environment
        :return: Initial observation after reset
        """
        _, info = self.env.reset(seed=seed, options=options)
        eeg_csv_file = select_random_eeg_mat_file()
        eeg_session = load_random_session_data(eeg_csv_file)[..., ::10]
        # Reshape session into samples of size window length
        mod = eeg_session.shape[-1] % self.window_length
        # (n_channels, recording_length) -> (n_samples, window_length, n_channels)
        self.eeg_data_windows = eeg_session[..., :-mod].reshape(
            eeg_session.shape[0], -1, self.window_length).transpose(1, 2, 0)
        
        self.__step__ = 0

        return self.__get_obs__(), info

    def step(self, action):
        """
        Take a step in the environment using the given action.
        :param action: Action to be taken
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        _, r, d, t, i = self.env.step(action)
        if self.__step__ is None:
            raise ValueError("Environment not reset. Call reset() before stepping.")
        self.__step__ += 1
        obs = self.__get_obs__()
        return obs, r, d, t, i
    
    def render(self):
        return self.env.sim.renderer.render_offscreen(camera_id=1)

    def close(self):
        self.env.env.close()


class EEGNonTdaEnv(gym.Env):
    """
    EEG-based Reinforcement Learning Environment using MyoSuite.
    """
    __step__: int = 0
    eeg_data_windows: Optional[np.ndarray] = None
    metadata = {"render_modes": []}

    def __init__(self, window_length: int = 10, eeg_channels: int = 64):
        """
        Initialize the EEG RL Environment.
        :param window_length: EEG signal window time length in ms
        :param n_filters: Number of filters used for TDA
        """
        super().__init__()
        self.env = gym.make("myoLegWalk-v0")
        self.action_space = self.env.action_space
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(window_length, eeg_channels), dtype=np.float32)
        self.window_length = window_length
        self.eeg_channels = eeg_channels

    def __get_obs__(self):
        """
        Get the current observation from the EEG data windows.
        :return: Current EEG observation
        """
        if self.__step__ is None:
            raise ValueError("Environment not reset. Call reset() before getting observations.")
        eeg_obs = self.eeg_data_windows[self.__step__]
        eeg_pos_enc = sinusoidal_positional_encoding(eeg_obs)
        return eeg_pos_enc

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        :param seed: Random seed for reproducibility
        :param options: Additional options for resetting the environment
        :return: Initial observation after reset
        """
        _, info = self.env.reset(seed=seed, options=options)
        eeg_csv_file = select_random_eeg_mat_file()
        eeg_session = load_random_session_data(eeg_csv_file)
        # Reshape session into samples of size window length
        mod = eeg_session.shape[-1] % self.window_length
        # (n_channels, recording_length) -> (n_samples, window_length, n_channels)
        self.eeg_data_windows = eeg_session[..., :-mod].reshape(
            eeg_session.shape[0], -1, 10).transpose(1, 2, 0)
        
        self.__step__ = 0

        return self.__get_obs__(), info

    def step(self, action):
        """
        Take a step in the environment using the given action.
        :param action: Action to be taken
        :return: Tuple of (observation, reward, done, truncated, info)
        """
        _, r, d, t, i = self.env.step(action)
        if self.__step__ is None:
            raise ValueError("Environment not reset. Call reset() before stepping.")
        self.__step__ += 1
        obs = self.__get_obs__()
        return obs, r, d, t, i
    
    def render(self):
        return self.env.sim.renderer.render_offscreen(camera_id=1)

    def close(self):
        self.env.env.close()

