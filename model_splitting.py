import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from copy import deepcopy
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

number_tasks = 3
input_dim = 2 * number_tasks
output_dim = input_dim
hidden_dim = 20 * input_dim
batch_size = 1

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linars1 = [nn.Linear(1, 1) for i in range(input_dim) * self.hidden_dim]
        self.linars2 = [nn.Linear(1, 1) for i in range(self.hidden1)]
        self.linars3 = [nn.Linear(1, 1) for i in range(self.hidden2)]
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # x = torch.tanh(self.fc4(x))
        coded = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(coded))
        # x = torch.tanh(self.fc7(x))
        # x = torch.tanh(self.fc8(x))
        x = torch.tanh(self.fc9(x))
        output = torch.sinh(self.fc10(x))
        return output, coded


x = torch.tensor([4,5,6,7,8], dtype=torch.float, requires_grad=True)
s = [elem for elem in x]
s2 = [elem**2 for elem in s]
print(s)
gg = torch.autograd.grad(s2[0], s[0])
print(gg)