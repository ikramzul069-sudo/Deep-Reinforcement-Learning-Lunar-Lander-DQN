# Deep Reinforcement Learning: Lunar Lander DQN

<div align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Environment-Gymnasium-green.svg" alt="Gymnasium">
</div>

---

## 1. About This Project & File Structure

### Project Description
This project implements a Deep Reinforcement Learning (DRL) agent to solve the navigation task in the LunarLander-v3 environment from the Gymnasium documentation. 

The primary objective of this project is to train an agent capable of landing a lunar module safely between two flags at coordinates (0,0) while minimizing fuel consumption. This is achieved by determining the optimal strategy for:
* **Deep Q-Network (DQN) Architecture:** Integrating traditional Q-Learning with Deep Neural Networks using an 8-unit input layer, 2 hidden layers with 64 neurons each (ReLU activation), and a 4-unit output layer representing the available engine actions.
* **Experience Replay:** Utilizing a memory buffer of 100,000 transitions to store experiences $(s, a, r, s', done)$ and randomly sample them to break sequential correlations.
* **Fixed Q-Targets & Soft Update:** Employing separate Policy (Local) and Target networks, updating the Target network gradually with a soft update parameter ($\tau = 0.001$) to prevent drastic oscillations[cite: 25, 26].
* **Exploration vs. Exploitation Balance:** Implementing an Epsilon-Greedy strategy where epsilon starts at 1.0 (100% exploration), decays at a rate of 0.995, and reaches a minimum of 0.01 (1% exploration) to ensure global optimization while achieving high scores.

### File Structure & Functionality
Here is a breakdown of what each file in this repository does:
* **`lunar_lander.py`**: The main Python script containing the complete DRL pipeline. It defines the `QNetwork` architecture, the `DQNAgent` logic, the training loop, and functions to plot results or record the final agent's performance.
* **`lunar_lander_model.pth`** *(Generated)*: The saved model weights containing the trained local Q-network state dictionary.
* **`plot.png`** *(Generated)*: The visualization plot illustrating the learning progress (rewards per episode) and the training loss over steps.

---

## 2. Set-Up
Before running this project, ensure your local environment has the following dependencies installed:

1. **Python:** Python 3.x installed on your system.
2. **Python Libraries:** Install the required reinforcement learning, deep learning, and plotting packages using `pip`:
```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

## 3. Compilation
Open your terminal in the project directory where lunar_lander.py is located to prepare for execution.

## 4. Execution
Run this simulation as a single standard Python script.

Running the Script:
Open your terminal and run the following command:
```Bash
python lunar_lander.py
```

Expected Results:
Once the training loop finishes executing its 1000 episodes, you should see the following outputs:

Console Output: The terminal will display the current episode and the rolling average score. The environment is considered officially solved once the average score over 100 consecutive episodes exceeds +200 , which this agent successfully achieves in 560 episodes

Saved Model Weights: A lunar_lander_model.pth file will be generated in your project directory containing the optimized network weights.

Performance Visualization: A plot.png file will be saved, showcasing a clear upward trend in rewards as the agent transitions from intensive exploration to expert landing , alongside a downward trend in training loss as the neural network minimizes prediction errors.
<img width="1200" height="500" alt="plot" src="https://github.com/user-attachments/assets/6705a0ef-5838-46e8-9465-158f29ebb8c0" />

