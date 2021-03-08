import torch
a = torch.empty(3, dtype=torch.long).random_(5)
b = torch.tensor([2,1,0])
print(a)
print(a[b])