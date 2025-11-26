# Frozen Lake Reinforcement Learning Implementation

## Project Structure

```
frozen_lake/
├── frozen_lake.py              # Main implementation
├── main.py                     # Entry point for running all algorithms
```

## Implementation Summary

This project implements a comprehensive reinforcement learning framework for the Frozen Lake environment, covering:

### 1. Environment Implementation
- **FrozenLake**: Grid-world environment with stochastic transitions
- Supports configurable slip probability (0.1 by default)
- Proper handling of terminal states (goal and holes)
- Absorbing state for episode termination

### 2. Model-Based Algorithms
- **Policy Evaluation**: Iterative computation of value function for a given policy
- **Policy Improvement**: Greedy policy extraction from value function
- **Policy Iteration**: Alternates between evaluation and improvement until convergence
- **Value Iteration**: Direct computation of optimal value function

### 3. Model-Free Algorithms (Tabular)
- **SARSA**: On-policy temporal difference learning
- **Q-Learning**: Off-policy temporal difference learning
- Both support ε-greedy exploration with decaying parameters

### 4. Linear Function Approximation
- **LinearWrapper**: Converts tabular environment to feature-based representation
- **Linear SARSA**: SARSA with linear function approximation
- **Linear Q-Learning**: Q-learning with linear function approximation
- Demonstrates that tabular methods are special cases of linear approximation

### 5. Deep Reinforcement Learning
- **FrozenLakeImageWrapper**: Converts states to 4-channel images
- **DeepQNetwork**: Convolutional neural network for Q-value estimation
- **ReplayBuffer**: Experience replay for stable training
- **DQN Learning**: Deep Q-Network algorithm with target network

## Key Features

### Correct Transition Probability
- Implements slip probability correctly: with probability (1-slip), move in desired direction; with probability slip, move randomly
- Handles boundary conditions (agent stays in place if moving out of bounds)
- Properly transitions to absorbing state at goal/hole

### Proper Reward Structure
- Reward of 1.0 only when taking action at goal state
- Reward of 0.0 for all other transitions
- Matches assignment specification exactly

### Convergence Criteria
- Policy iteration: Stops when policy doesn't change
- Value iteration: Stops when value changes < theta
- Model-free: Runs for fixed number of episodes

### Exploration Strategies
- ε-greedy with linear decay over episodes
- Proper tie-breaking in action selection (random among max Q-values)
- Separate learning rate and exploration factor decay

### PyTorch Dependency Handling
- Gracefully handles missing PyTorch
- DQN components only loaded when available
- Main script can run without PyTorch (skips DQN)

## Algorithm Details

### Policy Evaluation
```
V(s) = Σ_s' p(s'|s,π(s)) * [r(s'|s,π(s)) + γ * V(s')]
```

### Policy Improvement
```
π(s) = argmax_a Σ_s' p(s'|s,a) * [r(s'|s,a) + γ * V(s')]
```

### SARSA Update
```
Q(s,a) ← Q(s,a) + η * (r + γ * Q(s',a') - Q(s,a))
```

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + η * (r + γ * max_a' Q(s',a') - Q(s,a))
```

### Linear Approximation
```
Q(s,a) = φ(s,a)^T * θ
θ ← θ + η * δ * φ(s,a)
```

### DQN Loss
```
L = (r + γ * max_a' Q_target(s',a') - Q_online(s,a))^2
```

## Parameters

### Environment
- **slip**: 0.1 (10% chance of random action)
- **max_steps**: 16 (for 4×4 grid)
- **gamma**: 0.9 (discount factor)

### Model-Free Algorithms
- **max_episodes**: 4000
- **eta (learning rate)**: 0.5 → 0 (linear decay)
- **epsilon (exploration)**: 0.5 → 0 (linear decay)

### DQN
- **learning_rate**: 0.001
- **batch_size**: 32
- **buffer_size**: 256
- **target_update_frequency**: 4
- **kernel_size**: 3
- **conv_out_channels**: 4
- **fc_out_features**: 8

## Verification

All implementations have been verified against the reference specification:
- All 16 components correctly implemented
- Matches reference interface exactly
- Proper handling of edge cases
- Correct convergence criteria
- Proper exploration strategies

See `IMPLEMENTATION_EXPLANATION.md` for detailed verification.

## Usage

```python
from frozen_lake import FrozenLake, policy_iteration, value_iteration, sarsa, q_learning

# Create environment
lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)

# Run policy iteration
policy, value = policy_iteration(env, gamma=0.9, theta=0.001, max_iterations=128)
env.render(policy, value)

# Run Q-learning
policy, value, returns = q_learning(env, max_episodes=4000, eta=0.5, gamma=0.9, epsilon=0.5, seed=0)
env.render(policy, value)
```

## Notes

- The implementation uses NumPy for numerical computations
- PyTorch is optional (only needed for DQN)
- All algorithms properly handle the absorbing state
- Episode returns are tracked for model-free algorithms
- Linear approximation demonstrates that tabular methods are special cases
