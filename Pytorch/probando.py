## The usual imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.cuda.is_available()

## our data in tensor form
z = torch.cuda.FloatTensor([[-1.0],  [0.0], [1.0], [2.0], [3.0], [4.0]])
y = torch.cuda.FloatTensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]])
#x = torch.cuda.FloatTensor(-10, 10, 1000)
def sigmoid(a):
    return 1/(1+torch.exp(-a))

def derive(f):
    delta=0.001
    def grad_core(x):        
        return (f(x+delta)-f(x))/delta
    return grad_core

#plt.plot(z, sigmoid(z))
plt.plot(x, derive(sigmoid(x)))

plt.show()
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