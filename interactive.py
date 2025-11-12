from gridworld import GridWorld


def play_gridworld():
    """Interactive function to play the grid world game."""
    env = GridWorld(max_steps=100, seed=None)
    
    # Map user input to action indices
    # User presses: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
    # a c t i o n s = [ ’ 8 ’ , ’ 2 ’ , ’ 4 ’ , ’ 6 ’ ]
    # Action codes: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    actions = ['1', '2', '3', '4']
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    print("Grid World Game")
    print("===============")
    print("Controls: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT")
    print("Legend: A=Agent, G=Goal(+1), T=Trap(-1), #=Wall")
    print()
    
    state = env.reset()
    env.render()
    
    done = False
    total_reward = 0
    
    while not done:
        c = input('\nMove (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT): ')
        
        if c not in actions:
            print('Invalid action! Use 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT')
            continue
        
        # Map user input to action index: 1->0, 2->1, 3->2, 4->3
        action_idx = int(c) - 1
        state, reward, done = env.step(action_idx)
        
        print(f"\nAction: {action_names[action_idx]}")
        print(f"Reward: {reward}")
        total_reward += reward
        
        env.render()
        
        if done:
            print("\n" + "="*50)
            print("GAME OVER!")
            print("="*50)
            print(f"Total reward: {total_reward}")
            print(f"Steps taken: {env.n_steps}")
            if reward > 0:
                print("You reached the GOAL!")
            elif reward < 0:
                print("You fell into the TRAP!")
            print("="*50)


if __name__ == "__main__":
    play_gridworld()
