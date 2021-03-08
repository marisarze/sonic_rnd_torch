import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class base_net(nn.Module):
    def __init__(self, input_shape):
        super(base_net, self).__init__()
        self.name = 'base_net'
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 32, 6),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(32, 64, 4),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(64, 128, 4),
            nn.Tanh())

        self.num_features = self.get_flat_fts(input_shape, self.conv_part)
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.Tanh())


    def get_flat_fts(self, in_size, model):
        f = model(torch.ones(1, *in_size[-3:]))
        return int(np.prod(f.size()[1:]))

    def forward(self, state):
        x = state[:,-1,:,:,:]
        x.type(torch.FloatTensor)
        x = x/255.0 - 0.5
        x = self.conv_part(x)
        x = x.view(-1, self.num_features)
        x = self.fc1(x)
        return x

class bn_base_net(base_net):
    def __init__(self, input_shape):
        super(bn_base_net, self).__init__(input_shape)
        self.name = 'bn_base_net'
        self.conv_part = nn.Sequential(
            #nn.BatchNorm2d(3, momentum=0.0001),
            nn.Conv2d(3, 32, 6),
            nn.MaxPool2d(2),
            nn.Tanh(),
            #nn.BatchNorm2d(32, momentum=0.0001),
            nn.Conv2d(32, 64, 4),
            nn.MaxPool2d(2),
            nn.Tanh(),
            #nn.BatchNorm2d(64, momentum=0.01),
            nn.Conv2d(64, 64, 4),
            nn.Tanh())

        self.num_features = self.get_flat_fts(input_shape, self.conv_part)
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.Tanh())

class policy_net(nn.Module):
    def __init__(self, input_shape, action_space):
        super(policy_net, self).__init__()
        self.name = 'policy_net'
        self.base_part = bn_base_net(input_shape)
        self.end_part = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_space),
            nn.Softmax(-1))

    def forward(self, state):
        x = self.base_part(state)
        x = self.end_part(x)
        return x


class critic_net(nn.Module):
    def __init__(self, input_shape):
        super(critic_net, self).__init__()
        self.name = 'critic_net'
        self.base_part = bn_base_net(input_shape)
        self.last_layer = nn.Linear(512, 1)
        torch.nn.init.zeros_(self.last_layer.weight)
        torch.nn.init.zeros_(self.last_layer.bias)
        self.end_part = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            self.last_layer)

    def forward(self, state):
        x = self.base_part(state)
        x = self.end_part(x)
        return x


class reward_net(nn.Module):
    def __init__(self, input_shape):
        super(reward_net, self).__init__()
        self.name = 'reward_net'
        self.fast = self.create_branch(input_shape)
        self.slow = self.create_branch(input_shape)
        self.target = self.create_branch(input_shape)
        for param in self.target.parameters():
            param.requires_grad = False

    def create_branch(self, input_shape):
        return nn.Sequential(
            base_net(input_shape),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh())

    def forward(self, state):
        fast = self.fast(state)
        slow = self.slow(state)
        target = self.target(state)
        fast_diff = torch.mean(torch.abs(fast-target), dim=-1)
        slow_diff = torch.mean(torch.abs(slow-target), dim=-1)
        return fast_diff, slow_diff, fast_diff/slow_diff



if __name__=="__main__":
    from training_utils import *

    def policy_loss_fn(inputs, model, **kwargs):
        states = inputs[0]
        rewards = inputs[1]
        old_policies = inputs[2]
        crange = inputs[3]
        policy = model(states).double()
        policy = torch.min(policy, old_policies+crange)
        policy = torch.max(policy, old_policies-crange)
        #policy = torch.clip(policy, old_policies-crange, old_policies+crange)
        #beta= 1/2/crange
        base_loss = torch.sum(-rewards * policy, dim=-1)
        #inertia_loss = beta *  K.sum(K.abs(rewards), axis=-1) * K.sum(K.pow(policy-old_policies, 2), axis=-1)/self.action_space
        #loss = torch.sum(base_loss)  #+ inertia_loss)
        return base_loss

    def critic_loss_fn(inputs, model, **kwargs):
        states = inputs[0]
        targets = inputs[1]
        predicted = model(states)
        loss = torch.square(targets-predicted)
        return loss

    def reward_loss_fn(inputs, model, phase=0):
        states = inputs[0]
        target = inputs[1]
        initial = inputs[2]
        *pred, _ = model(states)
        loss = (pred[phase] - target) ** 2/ (initial+1e-11) ** 2
        return loss


    steps = 100
    width = 120
    height = 84
    action_space = 10
    state_shape = (1, 3, height, width)


    policy_model = policy_net(state_shape, action_space)
    critic_model = critic_net(state_shape)
    reward_model = reward_net(state_shape)

    states = np.ones((steps, *state_shape), dtype=np.uint8)
    cranges = np.ones((steps, action_space), dtype=np.float)
    rewards = np.random.rand(steps, action_space)
    old_policies = np.random.rand(steps, action_space)
    target_values = np.random.rand(steps, 1)

    # start = time.time()
    # optimizer = optim.Adam(policy_model.parameters(), lr=0.0001)
    # train_by_epoch([states, rewards, old_policies, cranges], policy_model, policy_loss_fn, optimizer)
    # print(time.time()-start)

    # start = time.time()
    # optimizer = optim.Adam(critic_model.parameters(), lr=0.0001)
    # train_by_epoch([states, target_values], critic_model, critic_loss_fn, optimizer)
    # print(time.time()-start)

    # start = time.time()
    # fast, slow, ratio = predict([states], reward_model, batch_size=2)
    # print(time.time()-start)

    # print(fast.shape, slow.shape, ratio.shape)
    # fast_diff = 5e-2 * fast
    # slow_diff = 5e-3 * slow
    # fast_target = fast - fast_diff
    # slow_target = slow - slow_diff
    # initial = [fast, slow]
    # diff = [fast_diff, slow_diff]
    # target = [fast_target, slow_target]

    # optimizer = optim.Adam(reward_model.parameters(), lr=0.0001)
    # start = time.time()
    # for phase in range(1):
    #     train_by_epoch([states, target[phase], initial[phase]], reward_model, reward_loss_fn, optimizer, phase=phase)
    # print(time.time()-start)

    # start = time.time()
    # for phase in range(1):
    #     train_by_batch([states, target[phase], initial[phase]], reward_model, reward_loss_fn, optimizer, phase=phase)
    # print(time.time()-start)

    # start = time.time()
    # results = predict([states], critic_model, verbose=True)
    # print(results)
    # print(results.shape, np.sum(results))
    # print(time.time()-start)

    start = time.time()
    results = evaluate([states, target_values], critic_model, critic_loss_fn, verbose=True)
    print(results)
    print(time.time()-start)



