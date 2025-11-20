import os
import sys
import csv
from typing import Dict, Any, List

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Make sure Python can find the 'environment' package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environment.custom_env import AgriScanEnv


def make_env():
    return AgriScanEnv(render_mode=None, max_steps=5)


def train_single_dqn(run_name: str, hyperparams: Dict[str, Any], total_timesteps: int = 50_000):
    print(f"\n=== Starting DQN run: {run_name} ===")
    print(f"Hyperparameters: {hyperparams}")

    env = DummyVecEnv([make_env])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        batch_size=hyperparams["batch_size"],
        buffer_size=hyperparams["buffer_size"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        target_update_interval=hyperparams["target_update_interval"],
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models/dqn", exist_ok=True)
    model_path = os.path.join("models", "dqn", f"dqn_{run_name}")
    model.save(model_path)
    print(f"Saved DQN model to: {model_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        render=False
    )
    print(f"Evaluation for {run_name}: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

    env.close()

    return {
        "run_name": run_name,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        **hyperparams,
    }


def main():
    os.makedirs("models/dqn", exist_ok=True)

    configs: List[Dict[str, Any]] = []

    learning_rates = [1e-4, 5e-4, 1e-3]
    gammas = [0.95, 0.98, 0.99]
    batch_sizes = [32, 64]
    exploration_fractions = [0.1, 0.2]
    base_buffer_size = 10_000
    target_update_intervals = [250, 500]

    run_index = 1
    for lr in learning_rates:
        for gamma in gammas:
            if run_index > 10:
                break
            configs.append(
                {
                    "run_name": f"run{run_index}_lr{lr}_g{gamma}",
                    "learning_rate": lr,
                    "gamma": gamma,
                    "batch_size": batch_sizes[run_index % len(batch_sizes)],
                    "buffer_size": base_buffer_size,
                    "exploration_fraction": exploration_fractions[run_index % len(exploration_fractions)],
                    "exploration_final_eps": 0.02,
                    "target_update_interval": target_update_intervals[run_index % len(target_update_intervals)],
                }
            )
            run_index += 1
        if run_index > 10:
            break

    results: List[Dict[str, Any]] = []

    for cfg in configs:
        run_name = cfg.pop("run_name")
        res = train_single_dqn(run_name, cfg, total_timesteps=50_000)
        res["run_name"] = run_name
        results.append(res)

    results_path = os.path.join("models", "dqn", "dqn_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(results_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSaved DQN results to: {results_path}")
    else:
        print("No DQN results to save.")


if __name__ == "__main__":
    main()
