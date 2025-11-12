import numpy as np
from environment import Environment


class CliffWalking(Environment):
    """git add 
    Cliff Walking environment (4x12 grid):
    - Agent starts at bottom-left
    - Goal is at bottom-right
    - Bottom row (except start and goal) is a cliff with -100 reward
    - Each step gives -1 reward (to encourage shorter paths)
    
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """
    
    def __init__(self, max_steps=200, seed=None):
        self.rows = 4
        self.cols = 12
        
        # States: grid positions + absorbing state
        n_states = self.rows * self.cols + 1
        n_actions = 4
        
        # Start position
        self.start_pos = (3, 0)
        self.start_state = self._pos_to_state(self.start_pos)
        
        # Goal position
        self.goal_pos = (3, 11)
        self.goal_state = self._pos_to_state(self.goal_pos)
        
        # Absorbing state
        self.absorbing_state = n_states - 1
        
        # Cliff positions (bottom row except start and goal)
        self.cliff_positions = [(3, col) for col in range(1, 11)]
        
        # Initial distribution: start at start position
        dist = np.zeros(n_states)
        dist[self.start_state] = 1.0
        
        super().__init__(n_states, n_actions, max_steps, dist, seed)
    
    def _state_to_pos(self, state):
        """Convert state number to (row, col) position."""
        if state == self.absorbing_state:
            return None
        row = state // self.cols
        col = state % self.cols
        return (row, col)
    
    def _pos_to_state(self, pos):
        """Convert (row, col) position to state number."""
        if pos is None:
            return self.absorbing_state
        row, col = pos
        return row * self.cols + col
    
    def _get_next_pos(self, pos, action):
        """Get next position given current position and action."""
        if pos is None:
            return None
        
        row, col = pos
        
        # Action effects: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        if action == 0:  # UP
            next_pos = (row - 1, col)
        elif action == 1:  # DOWN
            next_pos = (row + 1, col)
        elif action == 2:  # LEFT
            next_pos = (row, col - 1)
        elif action == 3:  # RIGHT
            next_pos = (row, col + 1)
        else:
            next_pos = pos
        
        # Check boundaries
        next_row, next_col = next_pos
        if (next_row < 0 or next_row >= self.rows or 
            next_col < 0 or next_col >= self.cols):
            return pos
        
        return next_pos
    
    def p(self, next_state, state, action):
        """Transition probability (deterministic)."""
        pos = self._state_to_pos(state)
        
        # If at goal, go to absorbing state
        if state == self.goal_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        # If in absorbing state, stay there
        if state == self.absorbing_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        # Get next position
        next_pos = self._get_next_pos(pos, action)
        
        # If fall off cliff, return to start
        if next_pos in self.cliff_positions:
            return 1.0 if next_state == self.start_state else 0.0
        
        expected_next_state = self._pos_to_state(next_pos)
        return 1.0 if next_state == expected_next_state else 0.0
    
    def r(self, next_state, state, action):
        """Reward function."""
        pos = self._state_to_pos(state)
        next_pos = self._state_to_pos(next_state)
        
        # If at goal, no more rewards
        if state == self.goal_state or state == self.absorbing_state:
            return 0.0
        
        # Check if moved to cliff
        if next_pos in self.cliff_positions or next_state == self.start_state:
            # Fell off cliff
            return -100.0
        
        # Check if reached goal
        if next_state == self.goal_state:
            return -1.0  # Still -1 for the step, but episode ends
        
        # Normal step cost
        return -1.0
    
    def step(self, action):
        """Override step to end episode when reaching goal."""
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        
        # End episode if we reach the goal or absorbing state
        if self.state == self.goal_state or self.state == self.absorbing_state:
            done = True
        
        return self.state, reward, done
    
    def render(self):
        """Render the current state of the environment."""
        print("\n" + "="*50)
        
        current_pos = self._state_to_pos(self.state)
        
        if current_pos is None:
            print("State: ABSORBING (Terminal)")
            print("="*50)
            return
        
        for row in range(self.rows):
            row_str = ""
            for col in range(self.cols):
                pos = (row, col)
                
                if pos == current_pos:
                    row_str += "[A]"
                elif pos == self.goal_pos:
                    row_str += "[G]"
                elif pos == self.start_pos:
                    row_str += "[S]"
                elif pos in self.cliff_positions:
                    row_str += "[C]"
                else:
                    row_str += "[ ]"
            print(row_str)
        
        print(f"\nState: {self.state}, Position: {current_pos}")
        print("="*50)


def play_cliff_walking():
    """Interactive function to play cliff walking."""
    env = CliffWalking(max_steps=200, seed=42)
    
    actions = ['8', '2', '4', '6']  # Numpad directions
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    print("Cliff Walking Game")
    print("==================")
    print("Controls: 8=UP, 2=DOWN, 4=LEFT, 6=RIGHT")
    print("Legend: A=Agent, G=Goal, S=Start, C=Cliff(-100)")
    print("Goal: Reach G without falling off the cliff!")
    print("Each step costs -1 reward")
    print()
    
    state = env.reset()
    env.render()
    
    done = False
    total_reward = 0
    
    while not done:
        c = input('\nMove (8/2/4/6): ')
        
        if c not in actions:
            print('Invalid action! Use 8(UP), 2(DOWN), 4(LEFT), or 6(RIGHT)')
            continue
        
        action_idx = actions.index(c)
        prev_state = state
        state, reward, done = env.step(action_idx)
        
        print(f"\nAction: {action_names[action_idx]}")
        print(f"Reward: {reward}")
        total_reward += reward
        
        if reward == -100:
            print("*** FELL OFF CLIFF! Returned to start ***")
        
        env.render()
        
        if state == env.goal_state or done:
            print(f"\nGame Over! Total reward: {total_reward}")
            print(f"Steps taken: {env.n_steps}")
            break


if __name__ == "__main__":
    play_cliff_walking()
