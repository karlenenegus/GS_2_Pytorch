import torch
import torch.nn as nn

class DNNGP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        linear_size = input_size
        self.input = nn.Identity()
        self.cnn1 = nn.Conv1d(in_channels = 1, out_channels=64, kernel_size=4)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(num_features=64)
        self.cnn2 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=4)
        self.drop2 = nn.Dropout(0.3)
        self.cnn3 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=4)
        linear_size = linear_size - 9
        self.dense = nn.Linear(in_features=linear_size*64, out_features=1)

    def forward(self, x):
        x = self.input(x)
        x = self.cnn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.batchnorm(x)
        x = self.cnn2(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.cnn3(x)
        x = self.act(x)
        x = torch.flatten(x, 1).unsqueeze(1)
        x = self.dense(x)
        
        return x