
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        temp = [np.atleast_1d(item[index]) for item in self.data]
        result = [torch.tensor(item, requires_grad=False) for item in temp]
        return result

    def __len__(self):
        return len(self.data[0])




def predict(inputs, model, batch_size=500, pin_memory=False, non_blocking=True, device=None, convert_type=None, verbose=False):
    model.eval()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory)

    check_loader = DataLoader(
                    dataset, 
                    batch_size=1,
                    pin_memory=pin_memory)
    check_batch = next(iter(check_loader))
    if convert_type:
        check_batch = [item.to(convert_type) for item in check_batch]
    check_pred = model(*(item.to(device, non_blocking=non_blocking) for item in check_batch))
    
    if isinstance(check_pred, tuple):
        scenario = 0
        num_items = len(check_pred)
    else:
        scenario = 1
    
    if scenario == 1:
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, ncols=130, disable=not verbose):
                batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
                if convert_type:
                    batch_input = [item.to(convert_type) for item in batch_input]
                pred = model(*batch_input)
                predictions.append(pred.cpu().numpy())
        predictions = np.concatenate(predictions)
    else:
        #predictions = [[]] * num_items
        predictions = []
        for ind in range(num_items):
            predictions.append([0])
        with torch.no_grad():
            for batch in tqdm(loader, ncols=130, disable=not verbose):
                batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
                if convert_type:
                    batch_input = [item.to(convert_type) for item in batch_input]
                pred = model(*batch_input)
                
                for j in range(num_items):
                    predictions[j].append(pred[j].cpu().numpy()[:])
        predictions = list(np.concatenate(elem[1:]) for elem in predictions)
    return predictions


def evaluate(inputs, model, loss_fn, batch_size=500, pin_memory=False, non_blocking=True, verbose=True, device=None, convert_type=None, **kwargs):
    model.eval()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )
    pbar = tqdm(loader, ncols=130, disable=not verbose)
    accumulated_loss = 0
    processed_samples = 0
    with torch.no_grad():
        for batch in pbar:
            batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
            if convert_type:
                batch_input = [item.to(convert_type) for item in batch_input]
            losses = loss_fn(batch_input, model, **kwargs)
            processed_samples += len(losses)
            loss = torch.sum(losses)
            accumulated_loss += loss.item()
            mean_loss = accumulated_loss/processed_samples
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_loss))
    return mean_loss


def train_by_batch(inputs, model, loss_fn, optimizer=None, batch_size=500, epochs=1, pin_memory=False, non_blocking=True, verbose=True, convert_type=None, **kwargs):
    model.train()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_history = []
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )
    for epoch in range(epochs):
        processed_samples = 0
        accumulated_epoch_loss = 0
        pbar = tqdm(loader, ncols=130, disable=not verbose)
        for batch in pbar:
            optimizer.zero_grad()
            batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
            if convert_type:
                batch_input = [item.to(convert_type) for item in batch_input]
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


def train_by_epoch(inputs, model, loss_fn, optimizer=None, batch_size=500, epochs=1, pin_memory=False, non_blocking=True, verbose=True, convert_type=None, **kwargs):
    model.train()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-6) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_history = []
    model = model.to(device)
    dataset = MyDataset(inputs)
    loader = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    pin_memory=pin_memory
                )
    for epoch in range(epochs):
        processed_samples = 0
        accumulated_epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(loader, ncols=130, disable=not verbose)
        for batch in pbar:
            batch_input = [item.to(device, non_blocking=non_blocking) for item in batch]
            if convert_type:
                batch_input = [item.to(convert_type) for item in batch_input]
            losses = loss_fn(batch_input, model, **kwargs)
            processed_samples += len(losses)
            loss = torch.sum(losses)
            accumulated_epoch_loss += loss.item()
            loss.backward()
            mean_epoch_loss = accumulated_epoch_loss/processed_samples
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= processed_samples
        optimizer.step()
        loss_history.append(mean_epoch_loss)
    return loss_history