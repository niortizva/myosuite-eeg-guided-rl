import os
import torch
import matplotlib.pyplot as plt
from eeg_rl.env import EEGTdaEnv
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
n_filters = 64

model_args = ModelArgsV1()
model_args.n_features = window_length
tda_env = EEGTdaEnv(
    window_length=window_length, eeg_channels=eeg_channels, n_filters=n_filters)
model = RomuloModel(
    tda_env.env.observation_space, 
    args=model_args, 
    features_dim=n_filters * window_length ** 2)

algorithms_tda = [
    {
       "name": "CrossQ",
        "algorithm": CrossQ,
        "args": [
            "MlpPolicy",
            tda_env,
        ],
        "kwargs": dict(
            features_extractor_class=RomuloModel,
            features_extractor_kwargs=dict(
                args=model_args, 
                features_dim=n_filters * window_length ** 2)
        )
    },
    {
       "name": "TD3",
        "algorithm": TD3,
        "args": [
            "MlpPolicy",
            tda_env,
        ],
        "kwargs": dict(
            features_extractor_class=RomuloModel,
            features_extractor_kwargs=dict(
                args=model_args, 
                features_dim=n_filters * window_length ** 2)
        )
    },
    {
       "name": "SAC",
        "algorithm": SAC,
        "args": [
            "MlpPolicy",
            tda_env,
        ],
        "kwargs": dict(
            features_extractor_class=RomuloModel,
            features_extractor_kwargs=dict(
                args=model_args, 
                features_dim=n_filters * window_length ** 2)
        )
    }
]


if __name__ == "__main__":
    iters = 1_000_000

    # Non TDA and TDA EEG Walking Training
    x = torch.randn((1, window_length, window_length, window_length))
    print(f"TDA Output shape: {model(x).shape}")

    print(f"Iterations: {iters}")

    for algorithm in algorithms_tda:
        name = algorithm["name"]
        print(f"TDA Training with algorithm: {name}")

        log_dir = f"tmp/TDA/{name}"
        os.makedirs(log_dir, exist_ok=True)

        alg = algorithm["algorithm"]
        policy, env = algorithm["args"]
        env = make_vec_env(
            EEGTdaEnv, 
            n_envs=8,
            seed=42, 
            monitor_dir=log_dir,
            env_kwargs=dict(
                 window_length=window_length, 
                 eeg_channels=eeg_channels, 
                 n_filters=n_filters
            )
        )

        kwargs = algorithm.get("kwargs", {})
        rl_model = alg(policy, env, policy_kwargs=kwargs, verbose=0)
        print(f"Initialized {alg.__name__} with model: {rl_model}")

        rl_model.learn(iters, progress_bar=True)

        rl_model.save(f"model/tda/{name}_tda_eeg_walking_model")

        plot_results(
            [log_dir], iters, results_plotter.X_TIMESTEPS, f"TDA EEG Walking - {name}")
        plt.savefig(f"model/tda/{name}_tda_eeg_walking_learning_curve.png")

