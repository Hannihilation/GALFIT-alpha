from torch import nn
import numpy as np


class GalfitAlpha(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, x):
        pass

    def fit(self, x, y):
        loss = nn.MSELoss()
        optimizer = nn.Adam(self.parameters(), lr=0.01)
        for i in range(1000):
            output = self(x)
            l = loss(output, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


class DeepQLearning:
    def __init__(self, learning_rate, reward_decay, e_greedy, memory_size, batch_size, state_dim, action_dim) -> None:
        self.eval_net = GalfitAlpha()
        self.target_net = GalfitAlpha()
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = np.zeros((memory_size, state_dim*2+2))
        self.memory_counter = 0

    def store_transition(self, s: np.ndarray, a, r, s_: np.ndarray):
        s = s.flatten()
        s_ = s_.flatten()
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next = self.target_net(batch_memory[:, -self.state_dim:])
        q_eval = self.eval_net(batch_memory[:, :self.state_dim])
        q_target = q_eval.copy()
        eval_act_index = batch_memory[:, self.state_dim].astype(int)
        reward = batch_memory[:, self.state_dim+1]
        q_target[:, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)
        self.eval_net.fit(batch_memory[:, :self.state_dim], q_target)

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.eval_net(s)
            action = np.argmax(action)
        return action

    def save(self):
        pass

    def load(self):
        pass

    def plot(self):
        pass
