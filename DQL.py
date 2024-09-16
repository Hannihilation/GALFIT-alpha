import matplotlib.pyplot as plt
from torch import nn, from_numpy
import numpy as np


class DeepQLearning:
    def __init__(self, Q_net: nn.Module, learning_rate, reward_decay, e_greedy, memory_size, batch_size) -> None:
        self.eval_net = Q_net
        self.target_net = Q_net.detach().clone()
        self.action_dim = Q_net.n_action
        self.state_dim = Q_net.state_num
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((memory_size, self.state_dim * 2 + 2))
        self.image_memory = np.zeros((memory_size, 2, Q_net.fig_channel_num,
                                     Q_net.fig_size, Q_net.fig_size))
        self.memory_counter = 0
        self.loss_history = []

    def store_transition(self, s: np.ndarray, a, r, s_: np.ndarray):
        transition = np.hstack((s[0], [a, r], s_[0]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.image_memory[index, 0, :, :, :] = s[1]
        self.image_memory[index, 1, :, :, :] = s_[1]
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        image_batch_memory = self.image_memory[sample_index, :, :, :, :]
        q_next = self.target_net(
            batch_memory[:, -self.state_dim:], image_batch_memory[:, 1, :, :, :])
        q_eval = self.eval_net(
            batch_memory[:, :self.state_dim], image_batch_memory[:, 0, :, :, :])
        q_target = q_eval.detach().clone()
        eval_act_index = batch_memory[:, self.state_dim].astype(int)
        reward = from_numpy(
            batch_memory[:, self.state_dim+1]).to(self.eval_net.device)
        q_target[:, eval_act_index] = reward + \
            self.gamma * q_next.detach().max(dim=1)[0]
        loss = self.eval_net.fit(
            batch_memory[:, :self.state_dim], image_batch_memory[:, 0, :, :, :], q_target, self.lr)
        self.loss_history.extend(loss)
        self.target_net = self.eval_net.detach().clone()

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            state_code = np.expand_dims(s[0], 0)
            figs = np.expand_dims(s[1], 0)
            action = self.eval_net(state_code, figs)
            action = np.argmax(action)
        return action

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
