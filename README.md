# AgriScan Reinforcement Learning Summative
1. Overview

This project applies Reinforcement Learning to a custom environment inspired by the AgriScan mission. AgriScan is an AI-powered tool that helps smallholder farmers diagnose crop diseases and nutrient deficiencies.

A simplified plant-health simulation was built using Gymnasium, and four RL algorithms were trained and compared:

DQN (Value-Based)

PPO (Policy Gradient)

A2C (Actor-Critic)

REINFORCE (Custom Implementation)

The goal is to learn optimal actions such as scanning, treating, or adjusting plant conditions.

2. Project Structure
   environment/
  custom_env.py
  rendering.py
training/
  dqn_training.py
  pg_training.py
models/
  dqn/
  pg/
plots/
main.py
analyze_results.py
plot_results.py
requirements.txt
README.md
3. Environment Description

The environment simulates plant conditions using a 4D observation space:

Disease level

Nutrient status

Moisture level

Environmental stress

Actions available:

Scan

Apply treatment

Adjust conditions

Do nothing

Rewards encourage correct diagnosis and penalize incorrect or unnecessary actions.

Visualization is provided through a Pygame interface.

4. Training

Training scripts:

dqn_training.py for DQN

pg_training.py for PPO, A2C, and REINFORCE

Each algorithm was trained under multiple hyperparameter combinations (10 runs each).
Model checkpoints and results are saved in models/.

5. Running the Best Agent

To run the best PPO agent with visualization:

python main.py

6. Analysis and Plots

To extract best hyperparameters:

python analyze_results.py


To generate performance plots:

python plot_results.py


Plots are saved in the plots/ directory.

7. Results Summary

Best-performing algorithms:

PPO: mean reward ≈ 10.55 (best overall)

DQN: mean reward ≈ 9.90

A2C: mean reward ≈ 9.70

REINFORCE: unstable and low performance

PPO showed the strongest stability and accuracy in decision-making.

8. Installation

Install dependencies:

pip install -r requirements.txt

9. Video Demonstration

A separate video demonstrates:

Environment overview

State, action, and reward structure

Training summary

Live execution of the best PPO agent with GUI visualization

10. License

Submitted as part of the Mission-Based Reinforcement Learning summative assessment.
