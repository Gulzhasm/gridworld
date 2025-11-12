"""Example usage of the reinforcement learning environments."""
from gridworld import GridWorld
from cliff_walking import CliffWalking


def gridworld_example():
    """Example of using GridWorld environment."""
    print("="*60)
    print("GRID WORLD EXAMPLE")
    print("="*60)
    
    # Create environment
    env = GridWorld(max_steps=50, seed=42)
    
    # Reset and display
    state = env.reset()
    print(f"\nInitial state: {state}")
    env.render()
    
    # Define a sequence of actions to reach the goal
    # Starting from state 2 (position 0,2), we go: RIGHT to reach goal
    actions = [3]  # RIGHT
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    print("\nExecuting action sequence to reach goal:")
    total_reward = 0
    
    for i, action in enumerate(actions):
        print(f"\nStep {i+1}: Taking action {action_names[action]}")
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        print(f"  Next state: {next_state}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        env.render()
        
        if next_state == env.goal_state:
            print("\nReached goal state! Taking one more action to get reward...")
            next_state, reward, done = env.step(action)
            total_reward += reward
            print(f"  Reward received: {reward}")
            print(f"  Moved to absorbing state: {next_state}")
            env.render()
            break
    
    print(f"\nTotal reward: {total_reward}")
    
    # Demonstrate transition probabilities
    print("\n" + "-"*60)
    print("TRANSITION PROBABILITIES EXAMPLE")
    print("-"*60)
    
    test_state = 0
    test_action = 3  # RIGHT
    print(f"\nFrom state {test_state} (position {env._state_to_pos(test_state)}), action RIGHT:")
    
    for ns in range(env.n_states):
        prob = env.p(ns, test_state, test_action)
        if prob > 0:
            pos = env._state_to_pos(ns)
            print(f"  -> State {ns} (position {pos}): probability = {prob}")
    
    # Demonstrate reward function
    print("\n" + "-"*60)
    print("REWARD FUNCTION EXAMPLE")
    print("-"*60)
    
    print(f"\nReward when taking action at goal state {env.goal_state}: {env.r(env.absorbing_state, env.goal_state, 0)}")
    print(f"Reward when taking action at trap state {env.trap_state}: {env.r(env.absorbing_state, env.trap_state, 0)}")
    print(f"Reward when taking action at normal state 0: {env.r(1, 0, 3)}")


def cliff_walking_example():
    """Example of using CliffWalking environment."""
    print("\n\n" + "="*60)
    print("CLIFF WALKING EXAMPLE")
    print("="*60)
    
    # Create environment
    env = CliffWalking(max_steps=100, seed=42)
    
    # Reset and display
    state = env.reset()
    print(f"\nInitial state: {state}")
    env.render()
    
    # Demonstrate falling off cliff
    print("\n" + "-"*60)
    print("FALLING OFF CLIFF")
    print("-"*60)
    
    print("\nTaking action RIGHT (will fall off cliff):")
    next_state, reward, done = env.step(3)  # RIGHT
    print(f"  Next state: {next_state}")
    print(f"  Reward: {reward}")
    print(f"  Returned to start: {next_state == env.start_state}")
    env.render()
    
    # Demonstrate safe path
    print("\n" + "-"*60)
    print("SAFE PATH (going around the cliff)")
    print("-"*60)
    
    env.reset()
    # Go up, then right along the safe path
    actions = [0, 3, 3, 3, 3, 3]  # UP, then multiple RIGHTs
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    print("\nTaking safe path:")
    total_reward = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: {action_names[action]} -> State {next_state}, Reward {reward}")
    
    env.render()
    print(f"\nTotal reward so far: {total_reward}")


if __name__ == "__main__":
    gridworld_example()
    cliff_walking_example()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
