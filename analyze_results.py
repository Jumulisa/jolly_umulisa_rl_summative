import csv
import os


def best_from_csv(path, algo_name):
    if not os.path.exists(path):
        print(f"[{algo_name}] File not found: {path}")
        return

    best_row = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean_r = float(row["mean_reward"])
            if (best_row is None) or (mean_r > float(best_row["mean_reward"])):
                best_row = row

    if best_row is None:
        print(f"[{algo_name}] No rows in {path}")
        return

    print(f"\n=== Best {algo_name} run ===")
    for k, v in best_row.items():
        print(f"{k}: {v}")


def main():
    best_from_csv("models/dqn/dqn_results.csv", "DQN")
    best_from_csv("models/pg/ppo_results.csv", "PPO")
    best_from_csv("models/pg/a2c_results.csv", "A2C")
    best_from_csv("models/pg/reinforce_results.csv", "REINFORCE")


if __name__ == "__main__":
    main()
