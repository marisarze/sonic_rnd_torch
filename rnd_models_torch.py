import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class base_net_torch(torch.nn.Module):
    def __init__(self, input_shape, action_space):
        super(base_net_torch, self).__init__()
        self.name = 'policy_net_torch'
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 32, 8),
            nn.MaxPool2d(4),
            nn.Tanh(),
            nn.Conv2d(32, 64, 4),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(64, 64, 4),
            nn.Tanh())
                    
        self.num_features = self.get_flat_fts(input_shape, self.conv_part)
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.Tanh(),
            nn.Linear(512, action_space),
            nn.Softmax(-1))


    def get_flat_fts(self, in_size, model):
        f = model(torch.ones(1, *in_size))
        return int(np.prod(f.size()[1:]))

    def forward(self, state):
        x = state[:,0,:,:,:]
        x.type(torch.FloatTensor)
        x = x/255.0 - 0.5
        x = self.conv_part(x)
        x = x.view(-1, self.num_features)
        x = self.fc1(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return [torch.from_numpy(item[index]) for item in self.data]

    def __len__(self):
        return len(self.data[0])


def predict_torch(model, device, inputs, batch_size=30, pin_memory=False, non_blocking=True, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = torch.utils.data.TensorDataset(*inputs)
    loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )

    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, ncols=130, disable=not verbose)):
            pred = model(*(item.to(device, non_blocking=non_blocking) for item in batch))
            pred = pred.cpu().numpy()
        predictions.append(pred)
    predictions = np.concatenate(predictions)
    return predictions
    
def calc_time(function, *args, **kwargs):
    print('-----------------------------------')
    print(f'calculating {function} with {kwargs}')
    start = time.time()
    for i in range(1):
        function(*args, **kwargs)
    print(time.time() - start)

def evaluate_torch(inputs, model, loss_fn, batch_size=32, pin_memory=False, non_blocking=True, verbose=True, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )
    pbar = tqdm(loader, ncols=130, disable=not verbose)
    for i, batch in enumerate(pbar):
        batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
        batch_loss = loss_fn(batch_input, model, **kwargs)
        accumulated_loss += batch_loss.numpy()
        mean_loss = accumulated_loss/end
        pbar.set_description('Evaluating {} '.format(model.name))
        pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_loss))
    return mean_loss


def train_by_batch_torch(inputs, model, loss_fn, optimizer, batch_size=32, epochs=1, pin_memory=False, non_blocking=True, verbose=True, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_history = []
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )
    for epoch in range(epochs):
        processed_samples = 0
        accumulated_epoch_loss = 0
        pbar = tqdm(loader, ncols=130, disable=not verbose)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
            losses = loss_fn(batch_input, model, **kwargs)
            processed_samples += len(losses)
            loss = torch.sum(losses)
            accumulated_epoch_loss += loss.item()
            torch.mean(losses).backward() 
            optimizer.step()
            mean_epoch_loss = accumulated_epoch_loss/processed_samples
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        loss_history.append(mean_epoch_loss)
    return loss_history

def train_by_epoch_torch(inputs, model, loss_fn, optimizer, batch_size=32, epochs=1, pin_memory=False, non_blocking=True, verbose=True, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_history = []
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )
    for epoch in range(epochs):
        processed_samples = 0
        accumulated_epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(loader, ncols=130, disable=not verbose)
        for i, batch in enumerate(pbar):
            batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
            losses = loss_fn(batch_input, model, **kwargs)
            processed_samples += len(losses)
            loss = torch.sum(losses)
            accumulated_epoch_loss += loss.item()
            loss.backward()
            mean_epoch_loss = accumulated_epoch_loss/processed_samples
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        for param in model.parameters():
            param.grad /= processed_samples
        optimizer.step()
        loss_history.append(mean_epoch_loss)
    return loss_history


def policy_loss_fn_tf(inputs, model, **kwargs):
        states = inputs[0]
        rewards = inputs[1]
        old_policies = inputs[2]
        crange = inputs[3]
        policy = model(states)
        policy = tf.clip_by_value(policy, old_policies-crange, old_policies+crange)
        #beta= 1/2/crange
        base_loss = K.sum(-rewards * policy, axis=-1)
        #inertia_loss = beta *  K.sum(K.abs(rewards), axis=-1) * K.sum(K.pow(policy-old_policies, 2), axis=-1)/self.action_space
        #+ inertia_loss)
        return base_loss

def policy_loss_fn_torch(inputs, model, **kwargs):
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

steps = 100000
width = 120
height = 84
action_space = 10
state_shape = (1, 3, height, width)
state_shape_tf = (1, height, width, 3)


mynet = base_net_torch(state_shape[1:], action_space)

states = np.ones((steps,*state_shape), dtype=np.uint8)
cranges = np.ones((steps, action_space), dtype=np.float)
rewards = np.random.rand(steps, action_space)
old_policies = np.random.rand(steps, action_space)

optimizer = optim.Adam(mynet.parameters(), lr=0.0001)


# start = time.time()
# train_by_batch_torch([states, rewards, old_policies, cranges], mynet, policy_loss_fn_torch, optimizer)
# print(time.time()-start)

# start = time.time()
# train_by_epoch_torch([states, rewards, old_policies, cranges], mynet, policy_loss_fn_torch, optimizer)
# print(time.time()-start)


import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from rnd_models import *
from train_util import *
mynet_tf = policy_net(state_shape_tf, action_space)

states = np.transpose(states, (0,1,3,4,2))


optimizer = Adam(lr=1e-6)
start = time.time()
train_by_batch_tf([states, rewards, old_policies, cranges], mynet_tf, policy_loss_fn_tf, optimizer)
print(time.time()-start)

start = time.time()
train_by_epoch_tf([states, rewards, old_policies, cranges], mynet_tf, policy_loss_fn_tf, optimizer)
print(time.time()-start)











#predict(mynet, device, [states])
#calc_time(predict, mynet, device, [states], batch_size=2048, pin_memory=False, non_blocking=True, verbose=True)



# import tensorflow as tf
# states = states.permute(0,1,3,4,2).numpy()
# print(states.shape)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# from rnd_models import *
# from train_util import *
# mynet_tf = policy_net(state_shape_tf, action_space)
# start = time.time()
# mynet_tf.predict(states, batch_size=2048, verbose=False)
# print(time.time() - start)





# for elem in loader:
#     print(type(elem), elem[0].shape)
