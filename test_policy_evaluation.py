import numpy as np
from gridworld import GridWorld
from policy_evaluation import evaluate_policy


def main():
    # Create grid world environment
    env = GridWorld(max_steps=100, seed=42)
    
    print("Grid World Environment:")
    print(f"- States: {env.n_states} (0-10: grid positions, 11: absorbing state)")
    print(f"- Actions: {env.n_actions} (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)")
    print(f"- Goal state: {env.goal_state} (position {env.goal_pos}) -> reward +1")
    print(f"- Trap state: {env.trap_state} (position {env.trap_pos}) -> reward -1")
    print(f"- Wall at position: {env.wall_pos}")
    print()
    
    # Define a deterministic policy
    policy = np.array([
        3,  # State 0 (0,0): RIGHT
        3,  # State 1 (0,1): RIGHT
        3,  # State 2 (0,2): RIGHT
        0,  # State 3 (0,3): UP (goal state)
        3,  # State 4 (1,0): RIGHT
        0,  # State 5 (1,1): UP (wall)
        0,  # State 6 (1,2): UP
        0,  # State 7 (1,3): UP (trap state)
        0,  # State 8 (2,0): UP
        0,  # State 9 (2,1): UP
        0,  # State 10 (2,2): UP
        0,  # State 11: absorbing state
    ])
    
    print("Testing Policy:")
    for s in range(env.n_states):
        pos = env._state_to_pos(s)
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        if pos:
            print(f"  State {s} {pos}: {action_names[policy[s]]}")
        else:
            print(f"  State {s} (absorbing): {action_names[policy[s]]}")
    print()
    
    # Run policy evaluation
    gamma = 0.9
    theta = 1e-6
    max_iterations = 1000
    
    print(f"Running policy evaluation with gamma={gamma}, theta={theta}")
    print("-" * 60)
    
    value = evaluate_policy(env, policy, gamma, theta, max_iterations)
    
    print("\nValue Function V^π:")
    print("-" * 60)
    for s in range(env.n_states):
        pos = env._state_to_pos(s)
        if pos:
            print(f"  State {s} {pos}: V = {value[s]:.4f}")
        else:
            print(f"  State {s} (absorbing): V = {value[s]:.4f}")


if __name__ == "__main__":
    main()
