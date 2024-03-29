import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvantageNetwork(nn.Module):
    """Advantage Network"""

    def __init__(self, input_size, output_size):
        super(AdvantageNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ValueNetwork(nn.Module):
    """Value Network"""

    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QNetwork(nn.Module):
    """Dueling Q-Network"""

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_channels = state_size[0]
        self.advantage_net = AdvantageNetwork(self.num_channels, action_size)
        self.value_net = ValueNetwork(self.num_channels)

    def forward(self, state):
        advantage = self.advantage_net(state)
        value = self.value_net(state)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
