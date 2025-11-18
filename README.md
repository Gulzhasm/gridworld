# Reinforcement Learning: Policy and Value Iteration

Implementation of reinforcement learning algorithms and environments for the Statistical Planning and Reinforcement Learning course exercises.

## Files

### Core Algorithms
- `policy_evaluation.py` - Policy evaluation, improvement, policy iteration, and value iteration algorithms

### Environments
- `environment.py` - Base classes (`MDPEnvironmentModel` and `SimulatedMDPEnvironment`) for MDP environments
- `gridworld.py` - Grid World environment (3×4 grid with goal, trap, and wall)

### Tests
- `test_policy_evaluation.py` - Test policy evaluation algorithm
- `test_all_algorithms.py` - Test policy evaluation, improvement, and policy iteration
- `test_value_iteration.py` - Compare value iteration vs policy iteration

## Algorithms

### Policy Evaluation
Computes the value function V^π for a given deterministic policy using the Bellman equation:
```
V(s) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
```
where a = π(s) is the action prescribed by the policy.

**Function**: `evaluate_policy(env, policy, gamma, theta, max_iterations)`

### Policy Improvement
Improves a policy by selecting actions that maximize the Q-value:
```
π'(s) = arg max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
```

**Function**: `improve_policy(env, policy, value, gamma)`

### Policy Iteration
Combines policy evaluation and improvement until convergence:
1. Evaluate current policy to get value function
2. Improve policy based on value function
3. Repeat until policy stabilizes

**Function**: `compute_policy_iteration(env, gamma, theta, max_iterations)`

### Value Iteration
Directly computes optimal value function using Bellman optimality equation:
```
V(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
```
Then extracts optimal policy from converged values.

**Function**: `compute_value_iteration(env, gamma, theta, max_iterations)`

## Grid World Environment

The Grid World is a 3×4 grid with:
- **Start position**: (2,2) - Agent always starts here
- **Goal state** at position (0,3): gives reward +1 when taking action at this state
- **Trap state** at position (1,3): gives reward -1 when taking action at this state
- **Wall** at position (1,1): blocks movement
- **Absorbing state**: entered after taking action at goal or trap (game ends immediately)
- **Actions**: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
- Invalid actions (hitting walls or boundaries) leave the state unchanged

### State Representation

States 0-10 represent grid positions (row-major order), state 11 is the absorbing state:
```
State layout:
[ 0][ 1][ 2][ 3-Goal]
[ 4][Wall][ 6][ 7-Trap]
[ 8][ 9][10-START][11]
```

Grid visualization:
```
[   ][   ][   ][ G ]
[   ][###][   ][ T ]
[   ][   ][ A ][   ]
```

## Usage

### Running Tests

```bash
# Test policy evaluation
python3 test_policy_evaluation.py

# Test all algorithms (evaluation, improvement, policy iteration)
python3 test_all_algorithms.py

# Compare value iteration vs policy iteration
python3 test_value_iteration.py
```

### Using Algorithms in Code

```python
from gridworld import GridWorldEnvironment
from policy_evaluation import evaluate_policy, improve_policy, compute_policy_iteration, compute_value_iteration

# Create environment
env = GridWorldEnvironment(max_steps=100, seed=42)

# Parameters
gamma = 0.9  # Discount factor
theta = 1e-6  # Convergence tolerance
max_iterations = 1000

# Policy Iteration
optimal_policy, optimal_value = compute_policy_iteration(env, gamma, theta, max_iterations)

# Value Iteration
optimal_policy, optimal_value = compute_value_iteration(env, gamma, theta, max_iterations)

# Manual policy evaluation
import numpy as np
policy = np.array([3, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0])
value = evaluate_policy(env, policy, gamma, theta, max_iterations)

# Policy improvement
improved_policy = improve_policy(env, policy, value, gamma)
```

## Interface Compliance

Environments implement the MDP interface:

### MDPEnvironmentModel
- `__init__(n_states, n_actions, seed)` - Constructor
- `p(next_state, state, action)` - Transition probability P(s'|s,a)
- `r(next_state, state, action)` - Reward R(s,a,s')
- `draw(state, action)` - Sample next state and reward

### SimulatedMDPEnvironment (extends MDPEnvironmentModel)
- `__init__(n_states, n_actions, max_steps, dist, seed)` - Constructor
- `reset()` - Reset environment and return initial state
- `step(action)` - Take action and return (next_state, reward, done)

## Algorithm Comparison

| Algorithm | Approach | Convergence | Use Case |
|-----------|----------|-------------|----------|
| Policy Iteration | Evaluate then improve | Guaranteed | When policy changes are infrequent |
| Value Iteration | Direct value optimization | Guaranteed | When you want optimal values directly |

Both algorithms converge to the same optimal policy and value function.

## Requirements

- Python 3.6+
- NumPy

## Notes

- States and actions are represented as integers starting from 0
- Transitions are deterministic in the Grid World
- All algorithms use in-place updates for efficiency
- Discount factor γ controls how much future rewards matter (0 = myopic, 1 = far-sighted)
