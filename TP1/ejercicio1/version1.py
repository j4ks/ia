dataset = [(1,6), (2,12), (3,18)]
        
def function(x):
    return x*x

def grad(f, x, delta=0.00000001):
    return (f(x+delta)-f(x))/delta

def linealModel(w,x):   #pendiente y punto
    return w*x

def error(x,y): #diferencia al cuadrado entre el valor estimado y el punto real (dataset).
    return (y-x)**2

def loss(m): #error total entre la funcion estimada y los puntos del dataset. (recibe pendiente estimada)
    sum = 0
    for index, (i, j) in enumerate(dataset):
        sum += error(linealModel(m,dataset[index][0]), dataset[index][1])
    return sum

def train(l=0.001, epsilon=0.0001):
    w = 0
    while(True):
        print(w)
        wprev = w
        w=w-l*grad(loss, w)
        if(abs(w-wprev)<epsilon):
            break
    return w
    
print(train())
    