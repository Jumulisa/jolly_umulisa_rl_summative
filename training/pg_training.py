import os
import sys
import csv
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environment.custom_env import AgriScanEnv


def make_env():
    return AgriScanEnv(render_mode=None, max_steps=5)


# =================== PPO ===================

def train_single_ppo(run_name: str, hyperparams: Dict[str, Any], total_timesteps: int = 50_000):
    print(f"\n=== Starting PPO run: {run_name} ===")
    print(f"Hyperparameters: {hyperparams}")

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        ent_coef=hyperparams["ent_coef"],
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models/pg", exist_ok=True)
    model_path = os.path.join("models", "pg", f"ppo_{run_name}")
    model.save(model_path)
    print(f"Saved PPO model to: {model_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, render=False
    )
    print(f"PPO Evaluation for {run_name}: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

    env.close()

    return {
        "run_name": run_name,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        **hyperparams,
    }


def run_ppo_experiments():
    os.makedirs("models/pg", exist_ok=True)

    configs: List[Dict[str, Any]] = []
    learning_rates = [1e-4, 3e-4, 1e-3]
    gammas = [0.95, 0.98, 0.99]
    n_steps_list = [64, 128]
    batch_sizes = [32, 64]
    ent_coefs = [0.0, 0.01]

    run_index = 1
    for lr in learning_rates:
        for gamma in gammas:
            if run_index > 10:
                break
            cfg = {
                "run_name": f"ppo_run{run_index}_lr{lr}_g{gamma}",
                "learning_rate": lr,
                "gamma": gamma,
                "n_steps": n_steps_list[run_index % len(n_steps_list)],
                "batch_size": batch_sizes[run_index % len(batch_sizes)],
                "ent_coef": ent_coefs[run_index % len(ent_coefs)],
            }
            configs.append(cfg)
            run_index += 1
        if run_index > 10:
            break

    results: List[Dict[str, Any]] = []
    for cfg in configs:
        run_name = cfg.pop("run_name")
        res = train_single_ppo(run_name, cfg, total_timesteps=50_000)
        res["run_name"] = run_name
        results.append(res)

    ppo_results_path = os.path.join("models", "pg", "ppo_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(ppo_results_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSaved PPO results to: {ppo_results_path}")
    else:
        print("No PPO results to save.")


# =================== A2C ===================

def train_single_a2c(run_name: str, hyperparams: Dict[str, Any], total_timesteps: int = 50_000):
    print(f"\n=== Starting A2C run: {run_name} ===")
    print(f"Hyperparameters: {hyperparams}")

    env = DummyVecEnv([make_env])

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        n_steps=hyperparams["n_steps"],
        ent_coef=hyperparams["ent_coef"],
        vf_coef=hyperparams["vf_coef"],
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models/pg", exist_ok=True)
    model_path = os.path.join("models", "pg", f"a2c_{run_name}")
    model.save(model_path)
    print(f"Saved A2C model to: {model_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, render=False
    )
    print(f"A2C Evaluation for {run_name}: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

    env.close()

    return {
        "run_name": run_name,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        **hyperparams,
    }


def run_a2c_experiments():
    os.makedirs("models/pg", exist_ok=True)

    configs: List[Dict[str, Any]] = []
    learning_rates = [1e-4, 3e-4, 1e-3]
    gammas = [0.95, 0.98, 0.99]
    n_steps_list = [5, 10]
    ent_coefs = [0.0, 0.01]
    vf_coefs = [0.5, 0.7]

    run_index = 1
    for lr in learning_rates:
        for gamma in gammas:
            if run_index > 10:
                break
            cfg = {
                "run_name": f"a2c_run{run_index}_lr{lr}_g{gamma}",
                "learning_rate": lr,
                "gamma": gamma,
                "n_steps": n_steps_list[run_index % len(n_steps_list)],
                "ent_coef": ent_coefs[run_index % len(ent_coefs)],
                "vf_coef": vf_coefs[run_index % len(vf_coefs)],
            }
            configs.append(cfg)
            run_index += 1
        if run_index > 10:
            break

    results: List[Dict[str, Any]] = []
    for cfg in configs:
        run_name = cfg.pop("run_name")
        res = train_single_a2c(run_name, cfg, total_timesteps=50_000)
        res["run_name"] = run_name
        results.append(res)

    a2c_results_path = os.path.join("models", "pg", "a2c_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(a2c_results_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSaved A2C results to: {a2c_results_path}")
    else:
        print("No A2C results to save.")


# =================== REINFORCE ===================

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action_and_log_prob(self, obs_np: np.ndarray) -> Tuple[int, torch.Tensor]:
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0)


def run_single_reinforce(
    run_name: str,
    hyperparams: Dict[str, Any],
    train_episodes: int = 500,
    eval_episodes: int = 10,
):
    print(f"\n=== Starting REINFORCE run: {run_name} ===")
    print(f"Hyperparameters: {hyperparams}")

    env = AgriScanEnv(render_mode=None, max_steps=5)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_size = hyperparams["hidden_size"]

    policy = PolicyNet(obs_dim, hidden_size, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=hyperparams["learning_rate"])
    gamma = hyperparams["gamma"]

    for episode in range(train_episodes):
        obs, info = env.reset()
        done = False

        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []

        while not done:
            action, log_prob = policy.get_action_and_log_prob(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss_terms = []
        for log_prob, Gt in zip(log_probs, returns):
            loss_terms.append(-log_prob * Gt)
        loss = torch.stack(loss_terms).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            ep_return = sum(rewards)
            print(f"REINFORCE {run_name} - Episode {episode + 1}/{train_episodes}, return={ep_return:.2f}")

    # Evaluation
    eval_returns = []
    for _ in range(eval_episodes):
        obs, info = env.reset()
        done = False
        total_r = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_r += reward
        eval_returns.append(total_r)

    mean_reward = float(np.mean(eval_returns))
    std_reward = float(np.std(eval_returns))
    print(f"REINFORCE Evaluation for {run_name}: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")

    os.makedirs("models/pg", exist_ok=True)
    model_path = os.path.join("models", "pg", f"reinforce_{run_name}.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"Saved REINFORCE model to: {model_path}")

    env.close()

    return {
        "run_name": run_name,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        **hyperparams,
    }


def run_reinforce_experiments():
    configs: List[Dict[str, Any]] = []

    learning_rates = [1e-3, 3e-3, 1e-2]
    gammas = [0.95, 0.98, 0.99]
    hidden_sizes = [32, 64]

    run_index = 1
    for lr in learning_rates:
        for gamma in gammas:
            if run_index > 10:
                break
            cfg = {
                "run_name": f"reinforce_run{run_index}_lr{lr}_g{gamma}",
                "learning_rate": lr,
                "gamma": gamma,
                "hidden_size": hidden_sizes[run_index % len(hidden_sizes)],
            }
            configs.append(cfg)
            run_index += 1
        if run_index > 10:
            break

    results: List[Dict[str, Any]] = []
    for cfg in configs:
        run_name = cfg.pop("run_name")
        res = run_single_reinforce(
            run_name,
            cfg,
            train_episodes=500,
            eval_episodes=10,
        )
        res["run_name"] = run_name
        results.append(res)

    reinforce_results_path = os.path.join("models", "pg", "reinforce_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(reinforce_results_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSaved REINFORCE results to: {reinforce_results_path}")
    else:
        print("No REINFORCE results to save.")


if __name__ == "__main__":
    run_ppo_experiments()
    run_a2c_experiments()
    run_reinforce_experiments()
