from torch import nn, optim, cat, from_numpy, float64


class GalfitAlpha(nn.Module):
    def __init__(self, state_num, fig_channel_num, fig_size, n_action, device) -> None:
        super(GalfitAlpha, self).__init__()
        self.state_num = state_num
        self.fig_channel_num = fig_channel_num
        self.fig_size = fig_size
        self.n_action = n_action
        # channel 0: residue, channel 1: components
        # CNN subgraph:
        kernel_size = fig_size // 4
        print(kernel_size)
        self.Conv1 = nn.Conv2d(in_channels=fig_channel_num,
                               out_channels=5, kernel_size=(5, 5), dtype=float64)
        self.Conv2 = nn.Conv2d(
            in_channels=5, out_channels=20, kernel_size=(kernel_size, kernel_size), dtype=float64)
        self.Conv3 = nn.Conv2d(
            in_channels=20, out_channels=10, kernel_size=(fig_size - 3 * kernel_size + 2, fig_size - 3 * kernel_size + 2), dtype=float64)
        # State all-connected subgraph:
        self.LL1 = nn.Linear(in_features=state_num,
                             out_features=5, dtype=float64)
        self.LL2 = nn.Linear(in_features=5, out_features=5, dtype=float64)
        self.LL3 = nn.Linear(in_features=5, out_features=5, dtype=float64)
        # Adding-up layer
        self.OutL1 = nn.Linear(
            in_features=15, out_features=30, dtype=float64)
        self.OutL2 = nn.Linear(
            in_features=30, out_features=20, dtype=float64)
        self.OutL3 = nn.Linear(
            in_features=20, out_features=n_action, dtype=float64)

        self.test_linear = nn.Linear(2, 5, dtype=float64)
        self.ReLU = nn.ReLU()

        self.device = device
        self.to(device)

    def forward_backup(self, state_code, figs):
        # ConvNN for figure:
        # channel 0: residue, channel 1: components
        state_code = from_numpy(state_code).to(self.device)
        figs = from_numpy(figs).to(self.device)
        print(figs.shape, state_code.shape)
        print(figs.dtype, state_code.dtype)
        x1 = self.ReLU(self.Conv1(figs))
        x1 = self.ReLU(self.Conv2(x1))
        x1 = self.ReLU(self.Conv3(x1))
        x1 = x1.view(x1.size(0), -1)
        # LinearNN for state code
        x2 = self.ReLU(self.LL1(state_code))
        x2 = self.ReLU(self.LL2(x2))
        x2 = self.ReLU(self.LL3(x2))
        # OutputNN
        x = cat(x1, x2, dim=1)
        x = self.ReLU(self.OutL1(x))
        x = self.ReLU(self.OutL2(x))
        x = self.OutL3(x)
        return x

    def forward(self, state_code, figs):
        x = from_numpy(state_code).to(self.device)
        x = self.test_linear(x)
        return x

    def fit(self, state_code, figs, y, lr):
        loss_his = []
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for i in range(1000):
            output = self(state_code, figs)
            l = loss(output, y)
            loss_his.append(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        return loss_his
