#Version 6: Multivariable

import statistics


class Lineal:

    weigh = 0
    weigh2 = 0
    bias = 0
    xDevi = 0
    yDevi = 0
    zDevi = 0
    
    def __init__(self, dataset, learningRate = 0.01 , epsilon=0.0001):
        self.dataset = dataset
        self.learningRate = learningRate
        self.epsilon = epsilon
        
    def updateDataset(self, dataset):
        self.dataset = dataset
        self.weigh = 0
        self.weigh2 = 0
        self.bias = 0
        
    def grad_w(self, f, w, w2, b):
        delta=0.000001
        return (f(w+delta, w2, b)-f(w, w2, b))/delta
    
    def grad_w2(self, f, w, w2, b):
        delta=0.000001
        return (f(w, w2+delta, b)-f(w, w2, b))/delta

    def grad_b(self, f, w, w2, b):
        delta=0.000001
        return (f(w, w2, b+delta)-f(w, w2, b))/delta

    def error(self,x,y): #diferencia al cuadrado entre el valor estimado y el punto real (dataset).
        return (y-x)**2

    def loss(self, w, w2,  b): #error total entre la funcion estimada y los puntos del dataset
        sum = 0
        for index, (i, j, k) in enumerate(self.dataset):
            sum += self.error((w*self.dataset[index][0]+w2*self.dataset[index][1])+b, self.dataset[index][2])
        return sum
    
    def normalize(self):
        xData = []
        zData = []
        
        for index, (i, j, k) in enumerate(self.dataset):
            xData.append(self.dataset[index][0]) #x
            zData.append(self.dataset[index][1]) #z
            
        xMean = statistics.mean(xData)
        self.xDevi = statistics.stdev(xData)
        zMean = statistics.mean(zData)
        self.zDevi = statistics.stdev(zData)

        for index, (i, j, k) in enumerate(self.dataset):
            self.dataset[index][0] = (self.dataset[index][0]-xMean)/self.xDevi
            self.dataset[index][1] = (self.dataset[index][1]-zMean)/self.zDevi

        print(self.dataset)
    
    def train(self):
        while(True):
            wprev = self.weigh
            w2prev = self.weigh2
            bprev = self.bias
            self.weigh=wprev-self.learningRate*self.grad_w(self.loss, self.weigh, w2prev, bprev)
            self.weigh2=w2prev-self.learningRate*self.grad_w2(self.loss, wprev, self.weigh2, bprev)
            self.bias=bprev-self.learningRate*self.grad_b(self.loss, wprev, w2prev, self.bias)
            if(abs(self.weigh-wprev)<self.epsilon and abs(self.bias-bprev)<self.epsilon and abs(self.weigh2-w2prev)<self.epsilon):
                break
        return self.weigh, self.weigh2,  self.bias


#Creando objetos y probando

set1 = [(6,2,40), (5,3,36), (4,10,44) ] # y=6x+2z
set2 = [(-5,-30), (-10,-55), (3,10), (10,45)] # y=5x-5
set3 = [[6,2,40],[5,3,36],[4,10,44],[-1,1,-4]]

prueba1 = Lineal(set3, 0.001, 0.0000001)

prueba1.normalize()

print(prueba1.train())

# Definir los números complejos
j = 1j
z1 = -1 + j
z2 = -2 - 2j

# Resolver la ecuación
alpha = z2 / z1

# Mostrar la solución
print("alpha = ", alpha)