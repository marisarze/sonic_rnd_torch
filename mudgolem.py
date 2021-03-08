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

number_tasks = 10
input_dim = 2 * number_tasks
output_dim = input_dim
hidden_dim = 5 * input_dim
batch_size = 1

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 6*6 from image dimension
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
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



def compute_fisher(model, loss):
    # flatten_params = torch.cat([layer.view(-1) for layer in model.parameters()])
    # print(flatten_params)
    grads1 = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    #shapes = [x.shape for x in grads1]
    flatten_grads1 = [x.view(-1) for x in grads1]
    parameters = [x for x in model.parameters()]

    #number_elements = [len(x) for x in flatten_grads1]
    flatten_grads2 = []
    for i in tqdm(range(len(flatten_grads1))):
        
        for j in range(len(flatten_grads1[i])):
            temp_grads = torch.autograd.grad(flatten_grads1[i][j], parameters[i], retain_graph=True)
            temp_grads = temp_grads[0].view(-1)
            flatten_grads2.append(temp_grads[j].view(-1))
    flatten_grads2 = torch.cat(flatten_grads2).detach()
    return flatten_grads2

def compute_fisher2(model, loss):
    
    def cat_tensors(tensors):
        t_tensor = torch.empty((sum(t.numel() for t in tensors),))
        offset = 0
        for t in tensors:
            t_tensor[offset:offset + t.numel()] = t.contiguous().view(-1)
            offset += t.numel()
        return t_tensor

    
    weights = list(model.parameters())
    loss_grad = torch.autograd.grad(loss, weights, create_graph=True, retain_graph=True)
    g_tensor = cat_tensors(loss_grad)
    w_tensor = cat_tensors(weights)
    
    l = g_tensor.size(0)
    print(g_tensor.size(0), w_tensor.size(0))
    hessian_diag = torch.empty(l)    
    for idx in range(l):
        print(g_tensor[idx], w_tensor[idx])
        grad2rd = torch.autograd.grad(g_tensor[idx], w_tensor[idx], create_graph=True, retain_graph=True)
        hessian_diag[idx] = grad2rd[0]
        

    return hessian_diag

def get_elastic_loss(model, reference, quadratic):
    params = torch.cat([x.view(-1) for x in model.parameters()])
    reference = torch.cat([x.view(-1) for x in reference])
    quadratic = torch.cat([x.view(-1) for x in quadratic])
    elastic = torch.zeros(params.shape, device=device)
    elastic = torch.sum(quadratic * torch.abs(params-reference))
    return elastic

def create_sample(task=None):
    x = torch.zeros((input_dim,), requires_grad=False, device=device)
    if task is not None:
        x[task] = 1.0
    else:
        x = x.random_(2)
    y = x.clone().detach()
    return x, y

def get_reference(net):
    reference = []
    for param in net.parameters():
        reference.append(param.data.view(-1))
    reference = torch.cat(reference)
    return reference

def get_base_loss(dataset, net):
    loss = 0
    j = 0
    for input, target in dataset:
        output, coded = net(input)
        loss += torch.sum((target - output) ** 2)
        j += 1
    loss /= j
    return loss


path = r'C:\Users\Andrey\Documents\pyexps\net.pth'
if os.path.exists(path):
    net = torch.load(path)
    print('net loaded')
else:
    net = Net()
    print('new net created')
net.to(device)

reference = get_reference(net)
quadratic = torch.zeros(reference.shape, device=device)
data = []
for i in range(number_tasks):
    elem = []
    elem.append(create_sample(2*i))
    elem.append(create_sample(2*i+1))
    data.append(elem)

optimizer = optim.Adam(net.parameters(), lr=0.001)
for task in range(number_tasks):
    base = get_base_loss(data[task], net)
    elastic = get_elastic_loss(net, reference, quadratic)
    while base>elastic+1e-5:
        losses = []
        for i in range(number_tasks):
            losses.append(get_base_loss(data[i], net).item())
        print('task', task, 'losses', losses)
        loss = base + elastic
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        base = get_base_loss(data[task], net)
        elastic = get_elastic_loss(net, reference, quadratic)
        
    

    quadratic += 0.5 * compute_fisher(net, base)
    reference = get_reference(net)
    for param in net.parameters():
        param.grad.zero_()

#torch.save(net, path)