from torch import nn, cat
import numpy as np
import copy
import matplotlib.pyplot as plt


class GalfitAlpha(nn.Module):
    def __init__(self, state_num, fig_channel_num, fig_size, n_action) -> None:
        # channel 0: residue, channel 1: components
        # CNN subgraph:
        kernel_size = np.floor(fig_size/4)
        self.Conv1 = nn.Conv2d(in_channels=fig_channel_num,
                               out_channels=5, kernel_size=2 * kernel_size)
        self.Conv2 = nn.Conv2d(
            in_channels=5, out_channels=20, kernel_size=kernel_size)
        self.Conv3 = nn.Conv2d(
            in_channels=20, out_channels=10, kernel_size=fig_size - 3 * kernel_size + 2)
        # State all-connected subgraph:
        self.LL1 = nn.Linear(in_features=state_num, out_features=5)
        self.LL2 = nn.Linear(in_features=5, out_features=5)
        self.LL3 = nn.Linear(in_features=5, out_features=5)
        # Adding-up layer
        self.OutL1 = nn.Linear(in_features=15, out_features=30)
        self.OutL2 = nn.Linear(in_features=30, out_features=20)
        self.OutL3 = nn.Linear(in_features=20, out_features=n_action)

        self.n_action = n_action

    def forward(self, state_code, figs):
        # ConvNN for figure:
        # channel 0: residue, channel 1: components
        x1 = nn.ReLU(self.Conv1(figs))
        x1 = nn.ReLU(self.Conv2(x1))
        x1 = nn.ReLU(self.Conv3(x1))
        x1 = x1.view(x1.size(0), -1)
        # LinearNN for state code
        x2 = nn.ReLU(self.LL1(state_code))
        x2 = nn.ReLU(self.LL2(x2))
        x2 = nn.ReLU(self.LL3(x2))
        # OutputNN
        x = cat(x1, x2)
        x = nn.ReLU(self.OutL1(x))
        x = nn.ReLU(self.OutL2(x))
        x = nn.Sigmoid(self.OutL3(x))
        return x

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
