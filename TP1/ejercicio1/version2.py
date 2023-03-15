# Versión 2, con bias.

dataset = [(1,10), (2,19), (3,28)]

def grad_w(f, w, b, delta=0.00000001):
    return (f(w+delta, b)-f(w, b))/delta

def grad_b(f, w, b, delta=0.00000001):
    return (f(w, b+delta)-f(w, b))/delta

def linealModel(w,x,b):   #pendiente y punto
    return w*x+b

def error(x,y): #diferencia al cuadrado entre el valor estimado y el punto real (dataset).
    return (y-x)**2

def loss(m,b): #error total entre la funcion estimada y los puntos del dataset. (recibe pendiente estimada)
    sum = 0
    for index, (i, j) in enumerate(dataset):
        sum += error(linealModel(m,dataset[index][0], b), dataset[index][1])
    return sum

def train(l=0.001, epsilon=0.000001):
    w = 0
    b = 0
    while(True):
        wprev = w
        bprev = b
        w=w-l*grad_w(loss, w, b)
        b=b-l*grad_b(loss, w, b)
        if(abs(w-wprev)<epsilon and abs(b-bprev)<epsilon):
            break
        
    return w,b
    
print(train())

