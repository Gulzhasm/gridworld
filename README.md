# Reinforcement Learning Environments

Implementation of reinforcement learning environments for the Statistical Planning and Reinforcement Learning course exercises.

## Files

- `environment.py` - Base classes (`EnvironmentModel` and `Environment`) following the specified interface
- `gridworld.py` - Grid World environment (3×4 grid with goal, trap, and wall)
- `cliff_walking.py` - Cliff Walking environment (4×12 grid)
- `interactive.py` - Interactive script to play Grid World
- `test_environments.py` - Test suite for the environments

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

## Cliff Walking Environment

A 4×12 grid where:
- **Start**: Bottom-left corner (3,0)
- **Goal**: Bottom-right corner (3,11)
- **Cliff**: Bottom row positions (3,1) through (3,10) - falling gives -100 reward and returns to start
- **Step cost**: -1 reward per step (encourages shorter paths)
- **Actions**: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

## Usage

### Running Tests

```bash
python test_environments.py
```

### Playing Grid World Interactively

```bash
python interactive.py
```

Controls:
- `1` - Move UP
- `2` - Move DOWN
- `3` - Move LEFT
- `4` - Move RIGHT

### Playing Cliff Walking Interactively

```bash
python cliff_walking.py
```

Same controls as Grid World.

### Using in Code

```python
from gridworld import GridWorld

# Create environment
env = GridWorld(max_steps=100, seed=42)

# Reset environment
state = env.reset()
env.render()

# Take actions
state, reward, done = env.step(action)

# Access transition probabilities
prob = env.p(next_state, state, action)

# Access rewards
reward = env.r(next_state, state, action)
```

## Interface Compliance

Both environments implement the required interface:

### EnvironmentModel
- `__init__(n_states, n_actions, seed)` - Constructor
- `p(next_state, state, action)` - Transition probability
- `r(next_state, state, action)` - Expected reward
- `draw(state, action)` - Sample next state and reward

### Environment (extends EnvironmentModel)
- `__init__(n_states, n_actions, max_steps, dist, seed)` - Constructor
- `reset()` - Reset environment and return initial state
- `step(action)` - Take action and return (next_state, reward, done)
- `render()` - Display current state (custom addition)

## Requirements

- Python 3.6+
- NumPy

## Notes

- States and actions are represented as integers starting from 0
- Transitions are deterministic in both environments
- The Grid World follows the exact specification from the lecture
- Cliff Walking is an additional environment for practice
# gridworld
