import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class the_model(nn.Module):
    def __init__(self):
        super(the_model, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.a * x**2 + self.b
        return x




net = the_model()
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss()



while True:
    input = torch.randn([1])
    target = 3 * input**2 + 5
    output = net(input)
    loss = criterion(output, target)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    for param in net.parameters():
        print(param.data.numpy())
    print('-------------------------------')


    torch.empty(4, 4).random_(2)