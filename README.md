# Deep Reinforcement Learning: Lunar Lander DQN

<div align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Environment-Gymnasium-green.svg" alt="Gymnasium">
</div>

---

## 1. About This Project & File Structure

### Project Description
[cite_start]This project implements a Deep Reinforcement Learning (DRL) agent to solve the navigation task in the LunarLander-v3 environment from the Gymnasium documentation[cite: 6]. 

[cite_start]The primary objective of this project is to train an agent capable of landing a lunar module safely between two flags at coordinates (0,0) while minimizing fuel consumption[cite: 7]. This is achieved by determining the optimal strategy for:
* [cite_start]**Deep Q-Network (DQN) Architecture:** Integrating traditional Q-Learning with Deep Neural Networks using an 8-unit input layer, 2 hidden layers with 64 neurons each (ReLU activation), and a 4-unit output layer representing the available engine actions[cite: 8, 9, 10, 12, 14].
* [cite_start]**Experience Replay:** Utilizing a memory buffer of 100,000 transitions to store experiences $(s, a, r, s', done)$ and randomly sample them to break sequential correlations[cite: 20, 21].
* [cite_start]**Fixed Q-Targets & Soft Update:** Employing separate Policy (Local) and Target networks, updating the Target network gradually with a soft update parameter ($\tau = 0.001$) to prevent drastic oscillations[cite: 25, 26].
* [cite_start]**Exploration vs. Exploitation Balance:** Implementing an Epsilon-Greedy strategy where epsilon starts at 1.0 (100% exploration), decays at a rate of 0.995, and reaches a minimum of 0.01 (1% exploration) to ensure global optimization while achieving high scores[cite: 35, 36, 37, 38].

### File Structure & Functionality
Here is a breakdown of what each file in this repository does:
* **`Assignment_2.py`**: The main Python script containing the complete DRL pipeline. It defines the `QNetwork` architecture, the `DQNAgent` logic, the training loop, and functions to plot results or record the final agent's performance.
* **`Report.pdf`**: The technical document detailing the implementation, hyperparameter choices, exploration vs. exploitation tradeoffs, differences between Multi-Armed Bandits and DRL, and a detailed discussion of training results[cite: 6, 29, 33, 40, 133].
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
Since this project uses Python, traditional compilation (like make or gcc) is not required. Instead, you utilize the Python interpreter to execute the script directly.

Open your terminal in the project directory where lunar_lander.py is located to prepare for execution.

## 4. Execution
Unlike an interactive notebook or a command-line executable, you run this simulation as a single standard Python script.

Running the Script:
Open your terminal and run the following command:

python lunar_lander.py
Expected Results:
Once the training loop finishes executing its 1000 episodes, you should see the following outputs:

Console Output: The terminal will display the current episode and the rolling average score. The environment is considered officially solved once the average score over 100 consecutive episodes exceeds +200 , which this agent successfully achieves in 560 episodes

Saved Model Weights: A lunar_lander_model.pth file will be generated in your project directory containing the optimized network weights.

Performance Visualization: A plot.png file will be saved, showcasing a clear upward trend in rewards as the agent transitions from intensive exploration to expert landing , alongside a downward trend in training loss as the neural network minimizes prediction errors.
