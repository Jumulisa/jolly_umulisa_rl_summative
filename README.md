# AgriScan Reinforcement Learning Summative


## 1. Overview
This project applies Reinforcement Learning to a custom environment inspired by the AgriScan mission.  
AgriScan aims to help smallholder farmers detect crop diseases early.  
A Gymnasium environment was created to simulate plant-health decision making, and four RL algorithms were trained and compared:
- DQN
- PPO
- A2C
- REINFORCE (custom)

The goal is to learn optimal actions such as scanning, treating disease, or adjusting conditions.



## 2. Project Structure
```
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
```



## 3. Environment Description
Observation Space (4 values):
- Disease level
- Nutrient status
- Moisture level
- Environmental stress

Action Space:
1. Scan plant  
2. Apply treatment  
3. Adjust conditions  
4. Do nothing  

Reward Structure:
- Correct treatment: +10  
- Correct adjustment: +6  
- Scan: +1  
- Doing nothing: -1  
- Incorrect decision: -4 to -10  

Rendering is implemented using Pygame.



## 4. Training
Training scripts:
```
python training/dqn_training.py
python training/pg_training.py
```

Algorithms trained:
- DQN (SB3)
- PPO (SB3)
- A2C (SB3)
- REINFORCE (manual PyTorch)

Each algorithm was trained with multiple hyperparameter combinations.  
Results saved under:
```
models/dqn/dqn_results.csv
models/pg/ppo_results.csv
models/pg/a2c_results.csv
models/pg/reinforce_results.csv
```



## 5. Running the Best Agent
```
python main.py
```
This loads the best PPO model and opens the Pygame visualization.



## 6. Analyzing Results
Extract best hyperparameters:
```
python analyze_results.py
```

Generate reward plots:
```
python plot_results.py
```

Plots saved in:
```
plots/
```



## 7. Results Summary
Best performing algorithms:
- PPO: mean reward ≈ 10.55  
- DQN: mean reward ≈ 9.90  
- A2C: mean reward ≈ 9.70  
- REINFORCE: unstable and low performance  

PPO achieved the most stable and optimal policy.



## 8. Installation
```
pip install -r requirements.txt
```



## 9. Video Demonstration Requirements
The required video includes:
- Explanation of environment
- Explanation of observation, action, and reward spaces
- Training summary
- Running the PPO agent using `python main.py`
- Showing terminal rewards and Pygame GUI
- Face camera on and full screen shared



## 10. License
Academic submission for the Mission-Based Reinforcement Learning Summative.  

