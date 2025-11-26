#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frozen_lake import (
    FrozenLake, policy_iteration, value_iteration, sarsa, q_learning,
    LinearWrapper, linear_sarsa, linear_q_learning, TORCH_AVAILABLE
)


def main():
    seed = 0
    
    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
    
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    
    print('# Model-based algorithms')
    print()
    
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)
    print()
    
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)
    print()
    
    print('# Model-free algorithms')
    max_episodes = 4000
    print()
    
    print('## Sarsa')
    policy, value, returns = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
                                    epsilon=0.5, seed=seed)
    env.render(policy, value)
    print()
    
    print('## Q-learning')
    policy, value, returns = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
                                         epsilon=0.5, seed=seed)
    env.render(policy, value)
    print()
    
    linear_env = LinearWrapper(env)
    print('## Linear Sarsa')
    # Unpack parameters and returns
    parameters, returns = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma,
                                       epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    print()
    
    print('## Linear Q-learning')
    # Unpack parameters and returns
    parameters, returns = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma,
                                            epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    print()
    
    if TORCH_AVAILABLE:
        from frozen_lake import FrozenLakeImageWrapper, deep_q_network_learning
        image_env = FrozenLakeImageWrapper(env)
        print('## Deep Q-network learning')
        # Unpack dqn and returns
        dqn, returns = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                               gamma=gamma, epsilon=0.2, batch_size=32,
                                               target_update_frequency=4, buffer_size=256,
                                               kernel_size=3, conv_out_channels=4,
                                               fc_out_features=8, seed=4)
        policy, value = image_env.decode_policy(dqn)
        image_env.render(policy, value)
    else:
        print('## Deep Q-network learning')
        print('PyTorch not available - skipping DQN')


if __name__ == '__main__':
    main()