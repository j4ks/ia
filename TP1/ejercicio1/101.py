def fun(x):
    return 6*x

dataset = [(1,6), (2,12), (3,18)]

print(dataset[0][1])

def grad(f, x, delta=0.001):
    return (f(x+delta)-f(x))/delta

def linealModel(w,x):
    return w*x

def error(x,y):
    return y-linealModel(x)

def loss(dataset, w, linealModel):
    sum = 0
    for i <
    return 0
    

def train(l=0.001, epsilon=0.000001):
    wprev = w
    while(true):
        w=w-l*grad(loss,w)
        if(w-wprev<epsilon):
            break
        print(w)
        return w
    

    