import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10,100)

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot(x)