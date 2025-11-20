import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths to CSV files
dqn_csv = "models/dqn/dqn_results.csv"
ppo_csv = "models/pg/ppo_results.csv"
a2c_csv = "models/pg/a2c_results.csv"
reinforce_csv = "models/pg/reinforce_results.csv"

# Output directory
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

def plot_results(csv_path, title, out_file):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["mean_reward"], marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Run Index")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, out_file))
    plt.close()

    print(f"Saved: {out_file}")

# Generate plots
plot_results(dqn_csv, "DQN Mean Rewards per Run", "dqn_rewards.png")
plot_results(ppo_csv, "PPO Mean Rewards per Run", "ppo_rewards.png")
plot_results(a2c_csv, "A2C Mean Rewards per Run", "a2c_rewards.png")
plot_results(reinforce_csv, "REINFORCE Mean Rewards per Run", "reinforce_rewards.png")

print("\nAll plots saved in the 'plots/' folder.")
