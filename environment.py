import numpy as np


class MDPEnvironmentModel:
    """Base class for Markov Decision Process environment models."""
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        """Transition probability P(s'|s,a)."""
        raise NotImplementedError()

    def r(self, next_state, state, action):
        """Reward function R(s,a,s')."""
        raise NotImplementedError()

    def draw(self, state, action):
        """Sample next state and reward from the environment model."""
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class SimulatedMDPEnvironment(MDPEnvironmentModel):
    """Simulated MDP environment for interactive episodes."""
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        MDPEnvironmentModel.__init__(self, n_states, n_actions, seed)
        self.max_steps = max_steps
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1.0 / n_states)

    def reset(self):
        """Reset environment to initial state."""
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        return self.state

    def step(self, action):
        """Execute one step in the environment."""
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done
