#Version 3: Orientado a Objetos y con bias.

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
        
    def grad_w(self, f, w, b):
        delta=0.00000001
        return (f(w+delta, b)-f(w,b))/delta
    
    def grad_b(self, f, w, b):
        delta=0.00000001
        return (f(w, b+delta)-f(w,b))/delta

    def error(self,x,y): #diferencia al cuadrado entre el valor estimado y el punto real (dataset).
        return (y-x)**2

    def loss(self, w, b): #error total entre la funcion estimada y los puntos del dataset
        sum = 0
        for index, (i, j) in enumerate(self.dataset):
            sum += self.error(w*self.dataset[index][0]+b, self.dataset[index][1])
        return sum

    def train(self):
        while(True):
            wprev = self.weigh
            bprev = self.bias
            self.weigh=wprev-self.learningRate*self.grad_w(self.loss, self.weigh, bprev)
            self.bias=bprev-self.learningRate*self.grad_b(self.loss, wprev, self.bias)
            if(abs(self.weigh-wprev)<self.epsilon and abs(self.bias-bprev)<self.epsilon):
                break
        return self.weigh, self.bias


#Creando objetos y probando

set1 = [(1,10), (2,19), (3,28), (6, 55)] # y=9x+1
set2 = [(-5,-30), (-10,-55), (3,10), (10,45)] # y=5x-5

prueba1 = Lineal(set1, 0.01, 0.0000001)

print(prueba1.train())