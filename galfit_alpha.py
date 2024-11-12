from torch import nn, optim, cat, float64
import numpy as np


class GalfitAlpha(nn.Module):
    def __init__(self, state_num, fig_channel_num, fig_size, n_action, device) -> None:
        super(GalfitAlpha, self).__init__()
        self.state_num = state_num
        self.fig_channel_num = fig_channel_num
        self.fig_size = fig_size
        self.n_action = n_action

        self.convs = nn.Sequential()
        layer_num = int(np.log2(fig_size))
        channel_now = fig_channel_num
        for i in range(layer_num):
            channel_next = channel_now * 2
            if channel_next > 16:
                channel_next = 16
            self.convs.add_module(f'conv{i}', nn.Conv2d(
                channel_now, channel_next, 3, 1, 1, device=device, dtype=float64))
            channel_now = channel_next
            self.convs.add_module(f'relu{i}', nn.ReLU())
            self.convs.add_module(f'maxpool{i}', nn.MaxPool2d(2, 2))

        self.linears = nn.Sequential(nn.Linear(state_num+channel_now, 32, device=device, dtype=float64),
                                     nn.ReLU(),
                                     nn.Linear(32, 64, device=device,
                                               dtype=float64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32, device=device,
                                               dtype=float64),
                                     nn.ReLU(),
                                     nn.Linear(32, n_action, device=device, dtype=float64))

        self.test_linear = nn.Linear(2, 5, dtype=float64)
        self.ReLU = nn.ReLU()

        self.device = device
        self.to(device)

    def forward(self, state_code, figs):
        # ConvNN for figure:
        # channel 0: residue, channel 1: components
        x = self.convs(figs)
        x = x.view(x.size(0), -1)
        x = cat((x, state_code), dim=1)
        x = self.linears(x)
        return x

    def forward_test(self, state_code, figs):
        x = self.test_linear(state_code)
        return x

    def fit(self, state_code, figs, y, lr):
        loss_his = []
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        print('start training ...')
        for i in range(1000):
            output = self(state_code, figs)
            l = loss(output, y)
            loss_his.append(l.cpu().detach().numpy())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'loss: {l}')
        return loss_his
