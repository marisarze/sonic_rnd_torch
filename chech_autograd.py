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

a = torch.tensor([1,2,3,4,5], dtype=torch.float, requires_grad=True).to(device)
b = torch.tensor([2,3,4,5,6], dtype=torch.float, requires_grad=True).to(device)
c = torch.sum(a + b * b, dim=-1).to(device)
print(c)
grads1 = torch.autograd.grad(c, b[0], retain_graph=True, create_graph=True)
print(grads1)

