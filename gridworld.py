import numpy as np
from environment import Environment


class GridWorld(Environment):
    """
    Grid world environment (3x4 grid):
    - 11 regular states (positions in grid)
    - 1 absorbing state (state 11)
    - 1 wall at position (1, 1)
    - Goal state at position (0, 3) -> gives reward +1
    - Trap state at position (1, 3) -> gives reward -1
    
    Actions: (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
    """
    
    def __init__(self, max_steps=100, seed=None):
        # 12 states total: 11 grid positions + 1 absorbing state
        # 4 actions: UP, DOWN, LEFT, RIGHT
        n_states = 12
        n_actions = 4
        
        # Grid dimensions
        self.rows = 3
        self.cols = 4
        
        # Special positions
        self.wall_pos = (1, 1)
        self.goal_pos = (0, 3)
        self.trap_pos = (1, 3)
        
        # Special states
        self.goal_state = 3  # position (0, 3)
        self.trap_state = 7  # position (1, 3)
        self.absorbing_state = 11
        
        # Initial distribution: always start at position (2, 2) which is state 10
        dist = np.zeros(n_states)
        dist[10] = 1.0  # Start at position (2, 2)
        
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
        if pos is None:  # Already in absorbing state
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
        
        # Check if next position is valid
        next_row, next_col = next_pos
        if (next_row < 0 or next_row >= self.rows or 
            next_col < 0 or next_col >= self.cols or
            next_pos == self.wall_pos):
            # Invalid move: stay in same position
            return pos
        
        return next_pos
    
    def p(self, next_state, state, action):
        """Transition probability."""
        # Deterministic transitions
        pos = self._state_to_pos(state)
        
        # If in goal or trap state, transition to absorbing state
        if state == self.goal_state or state == self.trap_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        # If in absorbing state, stay in absorbing state
        if state == self.absorbing_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        # Normal transition
        next_pos = self._get_next_pos(pos, action)
        expected_next_state = self._pos_to_state(next_pos)
        
        return 1.0 if next_state == expected_next_state else 0.0
    
    def r(self, next_state, state, action):
        """Reward function."""
        # Reward is given when taking action AT goal/trap state (not when moving into it)
        # This matches the specification: "receives reward upon taking an action at the goal state"
        if state == self.goal_state:
            return 1.0
        elif state == self.trap_state:
            return -1.0
        else:
            return 0.0
    
    def step(self, action):
        """Override step to end episode when reaching absorbing state."""
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        
        # End episode if we reach the absorbing state
        if self.state == self.absorbing_state:
            done = True
        
        return self.state, reward, done
    
    def render(self):
        """Render the current state of the environment."""
        print("\n" + "="*25)
        
        current_pos = self._state_to_pos(self.state)
        
        if current_pos is None:
            print("State: ABSORBING (Terminal)")
            print("="*25)
            return
        
        for row in range(self.rows):
            row_str = ""
            for col in range(self.cols):
                pos = (row, col)
                
                if pos == current_pos:
                    row_str += "[ A ]"
                elif pos == self.wall_pos:
                    row_str += "[###]"
                elif pos == self.goal_pos:
                    row_str += "[ G ]"
                elif pos == self.trap_pos:
                    row_str += "[ T ]"
                else:
                    row_str += "[   ]"
            print(row_str)
        
        print(f"\nState: {self.state}, Position: {current_pos}")
        print("="*25)
