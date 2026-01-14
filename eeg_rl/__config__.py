import os
from dataclasses import dataclass
from typing import List, Tuple


data_path: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data/ExperimentSessions")

files = ["S{i}_Data.mat" for i in range(1, 8)]


@dataclass
class ModelArgsV1:
    """
    Dataclass for Genesis model (Romulo) arguments version 1.
    """
    n_features: int = 64  # Number of features
    dim: int = 64  # Internal dimension of model
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 4  # Number of transformer layers
