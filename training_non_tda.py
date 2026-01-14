import os
import torch
import matplotlib.pyplot as plt
from eeg_rl.env import EEGNonTdaEnv
from eeg_rl.model import RomuloModel
from eeg_rl.__config__ import ModelArgsV1
# from stable_baselines3 import DDPG, SAC, TD3
from sbx import SAC, TD3, CrossQ
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.env_util import make_vec_env

window_length = 10
eeg_channels = 64
# n_filters = 64

model_args = ModelArgsV1()
model_args.n_features = eeg_channels
non_tda_env = EEGNonTdaEnv(
    window_length=window_length, eeg_channels=eeg_channels)
model = RomuloModel(
    non_tda_env.env.observation_space, 
    args=model_args, 
    features_dim=model_args.dim * window_length * eeg_channels)

algorithms_non_tda = [
    {
        "name": "CrossQ",
        "algorithm": CrossQ,
        "args": [
            "MlpPolicy",
            non_tda_env,
        ],
        "kwargs": dict(
            features_extractor_class=RomuloModel,
            features_extractor_kwargs=dict(
                args=model_args, 
                features_dim=window_length * eeg_channels)
        )
    },
    {
        "name": "TD3",
        "algorithm": TD3,
        "args": [
            "MlpPolicy",
            non_tda_env,
        ],
        "kwargs": dict(
            features_extractor_class=RomuloModel,
            features_extractor_kwargs=dict(
                args=model_args, 
                features_dim=window_length * eeg_channels)
        )
    },
    {
        "name": "SAC",
        "algorithm": SAC,
        "args": [
            "MlpPolicy",
            non_tda_env,
        ],
        "kwargs": dict(
            features_extractor_class=RomuloModel,
            features_extractor_kwargs=dict(
                args=model_args, 
                features_dim=window_length * eeg_channels)
        )
    }
]


if __name__ == "__main__":
    iters = 1_000_000

    # Non TDA and TDA EEG Walking Training
    x = torch.randn((1, window_length, eeg_channels))
    print(f"Non TDA Output shape: {model(x).shape}")

    print(f"Iterations: {iters}")

    for algorithm in algorithms_non_tda:
        name = algorithm["name"]
        print(f"Non TDA Training with algorithm: {name}")

        log_dir = f"tmp/NonTDA/{name}"
        os.makedirs(log_dir, exist_ok=True)

        alg = algorithm["algorithm"]
        policy, env = algorithm["args"]
        env = make_vec_env(
            EEGNonTdaEnv, 
            n_envs=8,
            seed=42, 
            monitor_dir=log_dir,
            env_kwargs=dict(
                 window_length=window_length, 
                 eeg_channels=eeg_channels
            )
        )

        kwargs = algorithm.get("kwargs", {})
        rl_model = alg(policy, env, policy_kwargs=kwargs, verbose=0)
        print(f"Initialized {alg.__name__} with model: {rl_model}")

        rl_model.learn(iters, progress_bar=True)

        rl_model.save(f"model/non_tda/{name}_non_tda_eeg_walking_model")

        plot_results(
            [log_dir], iters, results_plotter.X_TIMESTEPS, f"Non TDA EEG Walking - {name}")
        plt.savefig(f"model/non_tda/{name}_non_tda_eeg_walking_learning_curve.png")

