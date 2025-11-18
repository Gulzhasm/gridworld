import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    """
    Iterative policy evaluation (in-place).
    
    Args:
        env: Environment model with methods p(next_state, state, action) and r(next_state, state, action)
        policy: Deterministic policy as array where policy[s] = action for state s
        gamma: Discount factor
        theta: Convergence tolerance
        max_iterations: Maximum number of iterations
    
    Returns:
        value: Value function V^π for the given policy
    """
    value = np.zeros(env.n_states, dtype=np.float64)
    
    for iteration in range(max_iterations):
        delta = 0
        
        # For each state
        for s in range(env.n_states):
            v = value[s]
            
            # Get action from policy
            a = policy[s]
            
            # Compute new value: V(s) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
            new_value = 0.0
            for next_s in range(env.n_states):
                prob = env.p(next_s, s, a)
                reward = env.r(next_s, s, a)
                new_value += prob * (reward + gamma * value[next_s])
            
            value[s] = new_value
            delta = max(delta, abs(v - value[s]))
        
        # Check convergence
        if delta < theta:
            print(f"Policy evaluation converged after {iteration + 1} iterations (delta={delta:.6f})")
            break
    else:
        print(f"Policy evaluation stopped after {max_iterations} iterations (delta={delta:.6f})")
    
    return value


def policy_improvement(env, policy, value, gamma):
    """
    Policy improvement step.
    
    For each state, compute the action that maximizes the Q-value:
    π'(s) = arg max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
    
    Args:
        env: Environment model with methods p(next_state, state, action) and r(next_state, state, action)
        policy: Current deterministic policy as array where policy[s] = action
        value: Value function V^π for the current policy
        gamma: Discount factor
    
    Returns:
        improved_policy: Improved deterministic policy
    """
    improved_policy = np.zeros(env.n_states, dtype=int)
    
    # For each state
    for s in range(env.n_states):
        # Compute Q-value for each action
        action_values = np.zeros(env.n_actions)
        
        for a in range(env.n_actions):
            # Q(s, a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
            q_value = 0.0
            for next_s in range(env.n_states):
                prob = env.p(next_s, s, a)
                reward = env.r(next_s, s, a)
                q_value += prob * (reward + gamma * value[next_s])
            
            action_values[a] = q_value
        
        # Select action with maximum Q-value
        improved_policy[s] = np.argmax(action_values)
    
    return improved_policy


def policy_iteration(env, gamma, theta, max_iterations):
    """
    Policy iteration algorithm.
    
    Combines policy evaluation and policy improvement until convergence.
    
    Args:
        env: Environment model
        gamma: Discount factor
        theta: Convergence tolerance for policy evaluation
        max_iterations: Maximum number of policy iteration steps
    
    Returns:
        policy: Optimal deterministic policy
        value: Value function for the optimal policy
    """
    # Initialize policy randomly
    policy = np.zeros(env.n_states, dtype=int)
    value = np.zeros(env.n_states, dtype=np.float64)
    
    for iteration in range(max_iterations):
        # Policy evaluation
        value = policy_evaluation(env, policy, gamma, theta, max_iterations=1000)
        
        # Policy improvement
        improved_policy = policy_improvement(env, policy, value, gamma)
        
        # Check if policy has converged
        if np.array_equal(policy, improved_policy):
            print(f"Policy iteration converged after {iteration + 1} iterations")
            policy = improved_policy
            break
        
        policy = improved_policy
    else:
        print(f"Policy iteration stopped after {max_iterations} iterations")
    
    return policy, value


def value_iteration(env, gamma, theta, max_iterations):
    """
    Value iteration algorithm.
    
    Computes the optimal value function and policy by iteratively updating
    the value function using the Bellman optimality equation.
    
    Args:
        env: Environment model
        gamma: Discount factor
        theta: Convergence tolerance
        max_iterations: Maximum number of iterations
    
    Returns:
        policy: Optimal deterministic policy
        value: Optimal value function V*
    """
    value = np.zeros(env.n_states, dtype=np.float64)
    
    # Value iteration loop
    for iteration in range(max_iterations):
        delta = 0
        
        # For each state
        for s in range(env.n_states):
            v = value[s]
            
            # Compute max over all actions: V(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
            action_values = np.zeros(env.n_actions)
            
            for a in range(env.n_actions):
                # Q(s, a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
                q_value = 0.0
                for next_s in range(env.n_states):
                    prob = env.p(next_s, s, a)
                    reward = env.r(next_s, s, a)
                    q_value += prob * (reward + gamma * value[next_s])
                
                action_values[a] = q_value
            
            # Update value with maximum Q-value
            value[s] = np.max(action_values)
            delta = max(delta, abs(v - value[s]))
        
        # Check convergence
        if delta < theta:
            print(f"Value iteration converged after {iteration + 1} iterations (delta={delta:.6f})")
            break
    else:
        print(f"Value iteration stopped after {max_iterations} iterations (delta={delta:.6f})")
    
    # Extract optimal policy from converged value function
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        # Compute Q-value for each action
        action_values = np.zeros(env.n_actions)
        
        for a in range(env.n_actions):
            # Q(s, a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
            q_value = 0.0
            for next_s in range(env.n_states):
                prob = env.p(next_s, s, a)
                reward = env.r(next_s, s, a)
                q_value += prob * (reward + gamma * value[next_s])
            
            action_values[a] = q_value
        
        # Select action with maximum Q-value
        policy[s] = np.argmax(action_values)
    
    return policy, value
