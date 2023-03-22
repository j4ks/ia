## The usual imports
import torch
import torch.nn as nn

## our data in tensor form
x = torch.tensor([[-1.0],  [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float)
y = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float)


## Neural network with 1 hidden layer
layer1 = nn.Linear(1,1, bias=True)

## loss function
criterion = nn.L1Loss()

## optimizer algorithm
optimizer = torch.optim.SGD(layer1.parameters(), lr=0.01)

## training
for ITER in range(15100):
    layer1 = layer1.train()

    ## forward
    output = layer1(x)
    loss = criterion(output, y)
    optimizer.zero_grad()

    ## backward + update model params 
    loss.backward()
    optimizer.step()

    layer1.eval()
    print('Epoch: %d | Loss: %.4f' %(ITER, loss.detach().item()))
    
## test the model
sample = torch.tensor([10.0], dtype=torch.float)
predicted = layer1(sample)
print(predicted.detach().item())