import numpy as np
import contextlib
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@contextlib.contextmanager
def print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        self.max_steps = max_steps
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1.0 / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        n_states = self.lake.size + 1
        n_actions = 4
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.absorbing_state = n_states - 1
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def p(self, next_state, state, action):
        # 1. Absorbing State Dynamics
        if state == self.absorbing_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        # 2. Goal/Hole State Dynamics
        if self.lake_flat[state] in ['$', '#']:
             return 1.0 if next_state == self.absorbing_state else 0.0
        
        # 3. Normal Movement Dynamics
        row, col = np.unravel_index(state, self.lake.shape)
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)] # UP, LEFT, DOWN, RIGHT
        prob = 0.0
        
        for a in range(4):
            # Calculate slip probability
            if a == action:
                move_prob = 1.0 - self.slip + self.slip / 4.0
            else:
                move_prob = self.slip / 4.0
            
            dr, dc = directions[a]
            new_row, new_col = row + dr, col + dc
            
            # Boundary check
            if 0 <= new_row < self.lake.shape[0] and 0 <= new_col < self.lake.shape[1]:
                # Valid move: The new state is the grid index
                destination = np.ravel_multi_index((new_row, new_col), self.lake.shape)
            else:
                # Hit wall: stay put
                destination = state
            
            if destination == next_state:
                prob += move_prob
        
        return prob

    def r(self, next_state, state, action):
        # Reward is given when taking an action AT the goal state
        if state < self.lake.size and self.lake_flat[state] == '$':
            return 1.0
        return 0.0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            actions = ['^', '<', 'v', '>']
            print('Lake:')
            print(self.lake)
            print('Policy:')
            policy_arr = np.array([actions[a] for a in policy[:-1]])
            print(policy_arr.reshape(self.lake.shape))
            print('Value:')
            with print_options(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float64)
    
    for iteration in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            if s == env.absorbing_state:
                value[s] = 0
            else:
                a = policy[s]
                new_value = 0
                for ns in range(env.n_states):
                    p_ns = env.p(ns, s, a)
                    r_ns = env.r(ns, s, a)
                    new_value += p_ns * (r_ns + gamma * value[ns])
                value[s] = new_value
            delta = max(delta, abs(v - value[s]))
        
        if delta < theta:
            break
    
    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        if s == env.absorbing_state:
            policy[s] = 0
        else:
            action_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for ns in range(env.n_states):
                    p_ns = env.p(ns, s, a)
                    r_ns = env.r(ns, s, a)
                    action_values[a] += p_ns * (r_ns + gamma * value[ns])
            policy[s] = np.argmax(action_values)
    
    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    for iteration in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)
        
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float64)
    
    for iteration in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            if s == env.absorbing_state:
                value[s] = 0
            else:
                action_values = np.zeros(env.n_actions)
                for a in range(env.n_actions):
                    for ns in range(env.n_states):
                        p_ns = env.p(ns, s, a)
                        r_ns = env.r(ns, s, a)
                        action_values[a] += p_ns * (r_ns + gamma * value[ns])
                value[s] = np.max(action_values)
            delta = max(delta, abs(v - value[s]))
        
        if delta < theta:
            break
    
    policy = policy_improvement(env, value, gamma)
    return policy, value


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    returns = []
    
    for i in range(max_episodes):
        s = env.reset()
        
        # Tie-breaking action selection
        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            q_max = np.max(q[s])
            best_actions = np.flatnonzero(q[s] == q_max)
            a = random_state.choice(best_actions)
            
        episode_return = 0
        done = False
        
        while not done:
            ns, r, done = env.step(a)
            episode_return += r
            
            if done:
                q[s, a] += eta[i] * (r - q[s, a])
            else:
                if random_state.rand() < epsilon[i]:
                    na = random_state.choice(env.n_actions)
                else:
                    q_max = np.max(q[ns])
                    best_actions = np.flatnonzero(q[ns] == q_max)
                    na = random_state.choice(best_actions)
                
                q[s, a] += eta[i] * (r + gamma * q[ns, na] - q[s, a])
                s, a = ns, na
        
        returns.append(episode_return)
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value, returns


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    returns = []
    
    for i in range(max_episodes):
        s = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            if random_state.rand() < epsilon[i]:
                a = random_state.choice(env.n_actions)
            else:
                q_max = np.max(q[s])
                best_actions = np.flatnonzero(q[s] == q_max)
                a = random_state.choice(best_actions)
            
            ns, r, done = env.step(a)
            episode_return += r
            
            target = r
            if not done:
                target += gamma * np.max(q[ns])
                
            q[s, a] += eta[i] * (target - q[s, a])
            s = ns
        
        returns.append(episode_return)
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value, returns


class LinearWrapper:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    returns = []
    
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        
        # Tie-breaking action selection
        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            q_max = np.max(q)
            best_actions = np.flatnonzero(q == q_max)
            a = random_state.choice(best_actions)
            
        done = False
        episode_return = 0
        
        while not done:
            next_features, r, done = env.step(a)
            episode_return += r
            next_q = next_features.dot(theta)
            
            if random_state.rand() < epsilon[i]:
                next_a = random_state.choice(env.n_actions)
            else:
                nq_max = np.max(next_q)
                best_actions = np.flatnonzero(next_q == nq_max)
                next_a = random_state.choice(best_actions)
            
            delta = r + gamma * next_q[next_a] - q[a]
            theta += eta[i] * delta * features[a]
            
            features, q, a = next_features, next_q, next_a
        
        returns.append(episode_return)
    
    return theta, returns


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    returns = []
    
    for i in range(max_episodes):
        features = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            q = features.dot(theta)
            
            # Tie-breaking action selection
            if random_state.rand() < epsilon[i]:
                a = random_state.choice(env.n_actions)
            else:
                q_max = np.max(q)
                best_actions = np.flatnonzero(q == q_max)
                a = random_state.choice(best_actions)
            
            next_features, r, done = env.step(a)
            next_q = next_features.dot(theta)
            episode_return += r
            
            delta = r + gamma * np.max(next_q) - q[a]
            theta += eta[i] * delta * features[a]
            
            features = next_features
            
        returns.append(episode_return)
    
    return theta, returns


if TORCH_AVAILABLE:
    class FrozenLakeImageWrapper:
        def __init__(self, env):
            self.env = env
            lake = self.env.lake
            self.n_actions = self.env.n_actions
            self.state_shape = (4, lake.shape[0], lake.shape[1])
            
            lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]
            self.state_image = {
                self.env.absorbing_state: np.stack([np.zeros(lake.shape)] + lake_image)
            }
            
            for state in range(lake.size):
                row, col = np.unravel_index(state, lake.shape)
                agent_layer = np.zeros(lake.shape)
                agent_layer[row, col] = 1.0
                self.state_image[state] = np.stack([agent_layer] + lake_image)

        def encode_state(self, state):
            return self.state_image[state]

        def decode_policy(self, dqn):
            states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
            with torch.no_grad():
                q = dqn(states).detach().numpy()
            policy = q.argmax(axis=1)
            value = q.max(axis=1)
            return policy, value

        def reset(self):
            return self.encode_state(self.env.reset())

        def step(self, action):
            state, reward, done = self.env.step(action)
            return self.encode_state(state), reward, done

        def render(self, policy=None, value=None):
            self.env.render(policy, value)


    class DeepQNetwork(nn.Module):
        def __init__(self, env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed):
            nn.Module.__init__(self)
            torch.manual_seed(seed)
            
            self.conv_layer = nn.Conv2d(
                in_channels=env.state_shape[0],
                out_channels=conv_out_channels,
                kernel_size=kernel_size,
                stride=1
            )
            
            h = env.state_shape[1] - kernel_size + 1
            w = env.state_shape[2] - kernel_size + 1
            
            self.fc_layer = nn.Linear(
                in_features=h * w * conv_out_channels,
                out_features=fc_out_features
            )
            
            self.output_layer = nn.Linear(
                in_features=fc_out_features,
                out_features=env.n_actions
            )
            
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        def forward(self, x):
            x = torch.tensor(x, dtype=torch.float32)
            x = torch.relu(self.conv_layer(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc_layer(x))
            x = self.output_layer(x)
            return x

        def train_step(self, transitions, gamma, tdqn):
            states = np.array([t[0] for t in transitions])
            actions = np.array([t[1] for t in transitions])
            rewards = np.array([t[2] for t in transitions])
            next_states = np.array([t[3] for t in transitions])
            dones = np.array([t[4] for t in transitions])
            
            q = self(states)
            q = q.gather(1, torch.tensor(actions).view(len(transitions), 1).long())
            q = q.view(len(transitions))
            
            with torch.no_grad():
                next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)
                target = torch.tensor(rewards, dtype=torch.float32) + gamma * next_q
            
            loss = nn.MSELoss()(q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    class ReplayBuffer:
        def __init__(self, buffer_size, random_state):
            self.buffer = deque(maxlen=buffer_size)
            self.random_state = random_state

        def __len__(self):
            return len(self.buffer)

        def append(self, transition):
            self.buffer.append(transition)

        def draw(self, batch_size):
            indices = self.random_state.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]


    def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon, batch_size,
                                target_update_frequency, buffer_size, kernel_size, conv_out_channels,
                                fc_out_features, seed):
        random_state = np.random.RandomState(seed)
        replay_buffer = ReplayBuffer(buffer_size, random_state)
        dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed)
        tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed)
        epsilon = np.linspace(epsilon, 0, max_episodes)
        returns = []
        
        for i in range(max_episodes):
            state = env.reset()
            done = False
            episode_return = 0
            
            while not done:
                if random_state.rand() < epsilon[i]:
                    action = random_state.choice(env.n_actions)
                else:
                    with torch.no_grad():
                        q = dqn(np.array([state]))[0].numpy()
                        qmax = np.max(q)
                        best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                        action = random_state.choice(best)
                
                next_state, reward, done = env.step(action)
                episode_return += reward
                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                
                if len(replay_buffer) >= batch_size:
                    transitions = replay_buffer.draw(batch_size)
                    dqn.train_step(transitions, gamma, tdqn)
                
                if (i % target_update_frequency) == 0:
                    tdqn.load_state_dict(dqn.state_dict())
            
            returns.append(episode_return)
        
        return dqn, returns