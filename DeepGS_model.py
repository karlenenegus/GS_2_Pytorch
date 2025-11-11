import torch
import torch.nn as nn

class deepGS(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Identity()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout1d(p = .2)
        linear_size = ((input_size-21)/4)+1
        linear_size = int(linear_size - (linear_size % 1))
        self.fc1 = nn.Linear(in_features=(linear_size*8), out_features=32)
        self.drop2 = nn.Dropout1d(p = 0.1)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.act2 = nn.Sigmoid()
        #self.drop3 = nn.Dropout1d(p = 0.05) ##Used in original paper - performs better without.

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1).unsqueeze(1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x) 
        x = self.fc2(x)
        
        return x