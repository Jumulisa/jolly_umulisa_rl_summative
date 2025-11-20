from environment.custom_env import AgriScanEnv
from stable_baselines3 import DQN, PPO, A2C


def run_random_agent(num_episodes: int = 3):
    """
    Simple baseline: agent takes random actions.
    Useful for testing the environment + visualization.
    """
    env = AgriScanEnv(render_mode="human", max_steps=5)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"[RANDOM] Episode {ep + 1} finished with total reward: {total_reward:.2f}")

    env.close()


def run_best_dqn(num_episodes: int = 5):
    """
    Run the best DQN model (from dqn_results.csv).
    """
    model_path = "models/dqn/dqn_run1_lr0.0001_g0.95"
    print(f"Loading DQN model from: {model_path}.zip")

    env = AgriScanEnv(render_mode="human", max_steps=5)
    model = DQN.load(model_path)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        print(f"[DQN] Episode {ep + 1} finished with total reward: {total_reward:.2f}")

    env.close()


def run_best_ppo(num_episodes: int = 5):
    """
    Run the best PPO model (overall best agent).
    From analyze_results.py: ppo_run4_lr0.0003_g0.95
    """
    model_path = "models/pg/ppo_ppo_run4_lr0.0003_g0.95"
    print(f"Loading PPO model from: {model_path}.zip")

    env = AgriScanEnv(render_mode="human", max_steps=5)
    model = PPO.load(model_path)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        print(f"[PPO] Episode {ep + 1} finished with total reward: {total_reward:.2f}")

    env.close()


def run_best_a2c(num_episodes: int = 5):
    """
    Run the best A2C model.
    From analyze_results.py: a2c_run5_lr0.0003_g0.98
    """
    model_path = "models/pg/a2c_a2c_run5_lr0.0003_g0.98"
    print(f"Loading A2C model from: {model_path}.zip")

    env = AgriScanEnv(render_mode="human", max_steps=5)
    model = A2C.load(model_path)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        print(f"[A2C] Episode {ep + 1} finished with total reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # For your final demo/video, use the best overall agent:
    run_best_ppo(num_episodes=5)

    # For testing other agents:
    # run_best_dqn(num_episodes=5)
    # run_best_a2c(num_episodes=5)
    # run_random_agent(num_episodes=3)
