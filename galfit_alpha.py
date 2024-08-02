from torch import nn
import numpy as np
import copy
import matplotlib.pyplot as plt


class GalfitAlpha(nn.Module):
    def __init__(self, x_dim, y_dim, n_action) -> None:
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_action = n_action

    def forward(self, x):
        pass

    def fit(self, x, y, lr):
        loss_his = []
        loss = nn.MSELoss()
        optimizer = nn.Adam(self.parameters(), lr=lr)
        for i in range(1000):
            output = self(x)
            l = loss(output, y)
            loss_his.append(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        return loss_his


class DeepQLearning:
    def __init__(self, Q_net: nn.Module, learning_rate, reward_decay, e_greedy, memory_size, batch_size) -> None:
        self.eval_net = Q_net
        self.target_net = copy.deepcopy(Q_net)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_dim = Q_net.x_dim * Q_net.y_dim * 2
        self.memory = np.zeros((memory_size, self.state_dim*2+2))
        self.memory_counter = 0
        self.loss_history = []

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
        loss = self.eval_net.fit(
            batch_memory[:, :self.state_dim], q_target, self.lr)
        self.loss_history.extend(loss)
        # loss = nn.MSELoss()
        # optimizer = nn.Adam(self.eval_net.parameters(), lr=self.lr)
        # l = loss(q_eval, q_target)
        # optimizer.zero_grad()
        # l.backward()
        # optimizer.step()

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.eval_net(s)
            action = np.argmax(action)
        return action

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
