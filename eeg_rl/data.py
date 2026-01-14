import os
from random import choice
from .__config__ import data_path, files
import h5py
from numpy import (
    ndarray, arange, zeros, sin, cos, newaxis, dot, diag,
    sqrt, sort, unique, array_split, max as np_max, array)


def select_random_eeg_mat_file() -> str:
    """
    Select a random EEG MAT file from the CSV data directory.
    :return: Path to the randomly selected CSV file.
    """
    mat_files = os.listdir(data_path)
    random_csv_file_path: str = os.path.join(
        data_path, choice(mat_files))
    return random_csv_file_path


def load_random_session_data(random_csv_file_path: str) -> ndarray:
    """
    Load EEG session data from a random CSV file.
    :param random_csv_file_path: Path to the CSV file.
    :return: Numpy array containing the EEG session data.
    """
    with h5py.File(random_csv_file_path, 'r') as file:
        sessions = list(file["SessionData"].keys())
        random_session = choice(sessions)
        try:
            eeg_data = array(
                file["SessionData"][random_session]["trialData"]["SixMinWalk"]
                ["EEG"][:])
        except:
            return load_random_session_data(random_csv_file_path)

    return eeg_data


def sinusoidal_positional_encoding(
        sliding_window: ndarray, min_freq: float = 1e-4) -> \
        ndarray:
    """
    Generate sinusoidal positional encoding for the given sliding window.
    :param sliding_window: DataFrame representing the sliding window of EEG
    data.
    :param d_model: Dimension of the model (number of features).
    :param min_freq: Minimum frequency for the positional encoding.
    :return: Numpy array containing the positional encoding.
    """
    time_steps, n_features = sliding_window.shape
    positions = arange(time_steps)
    dimensions = arange(n_features)

    # Calculate angles
    angle_rates = 1 / (min_freq ** (2 * (dimensions // 2) / n_features))
    angle_rads = positions[:, newaxis] * angle_rates[newaxis, :]

    pos_encoding = zeros((time_steps, n_features))
    pos_encoding[:, 0::2] = sin(angle_rads[:, 0::2])  # even indices
    pos_encoding[:, 1::2] = cos(angle_rads[:, 1::2])  # odd indices

    return sliding_window + pos_encoding


def distance_matrix(sliding_window: ndarray) -> ndarray:
    """
    Compute the distance matrix for the given sliding window.
    :param sliding_window: Numpy array representing the sliding window of EEG
    data.
    :return: Numpy array containing the Euclidean distance matrix.
    """
    G = dot(sliding_window, sliding_window.T)
    D = diag(G).reshape(-1, 1)
    D_sq = D + D.T - 2 * G
    return sqrt(D_sq)


def simplex_adjacency_history(distances: ndarray, n_filters: int) -> ndarray:
    """
    Create a simplex adjacency history.
    :param distances: Numpy array representing the distance matrix.
    :param n_filters: Number of filters (size of the filtration).
    :return: Numpy array representing the simplex adjacency history.
    """
    unique_values = sort(distances.flatten())
    filters_values = array([
        np_max(f) for f in array_split(unique_values, n_filters)])[:, newaxis, newaxis]
    A = (distances <= filters_values).astype(float)
    return A

