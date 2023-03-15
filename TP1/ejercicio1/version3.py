#Version 2: Orientado a Objetos y con bias.

class Lineal:

    weigh = 0
    bias = 0
    
    def __init__(self, dataset, learningRate = 0.001 , epsilon=0.0001):
        self.dataset = dataset
        self.learningRate = learningRate
        self.epsilon = epsilon
        
    def updateDataset(self, dataset):
        self.dataset = dataset
        self.weigh = 0
        self.bias = 0
        
    def grad_w(f, x, bi):
        delta=0.00000001
        return (f(x+delta, bi)-f(x,bi))/delta

    def error(x,y): #diferencia al cuadrado entre el valor estimado y el punto real (dataset).
        return (y-x)**2

    def loss(self, m, b): #error total entre la funcion estimada y los puntos del dataset
        sum = 0
        for index, (i, j) in enumerate(self.dataset):
            sum += self.error(m*self.dataset[index][0]+b, self.dataset[index][1])
        return sum

    def train(self):
        while(True):
            wprev = self.weigh
            bprev = self.bias
            self.weigh=self.weigh-self.learningRate*self.grad(self.loss, self.weigh)
            self.bias=self.bias-self.learningRate*self.grad(self.loss, self.bias)
            if(abs(self.weigh-wprev)<self.epsilon and abs(self.bias-bprev)<self.epsilon):
                break
        print("Weigh= ", self.weigh, " Bias= ", self.bias)


#Creando clase y probando

set1 = [(1,10), (2,19), (3,28)] # y=9x+1

prueba1 = Lineal(set1, 0.001, 0.0001)

prueba1.train()