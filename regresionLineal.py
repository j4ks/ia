import random
import numpy as np

x = np.array([1,2,3])
y = np.array([2,5,8])



def model (x,w,b):    #Modelo lineal (input, weight, bias)
    return w*x+b

def error(x,y,w,b):   #Devuelve matriz de diferencias
    return model (x,w,b)-y

def loss(x,y,w,b):    #Desviacion total con respecto a la curva esperada
    e = error(x,y,w,b)
    return e@e      # devuelve un numero = producto matricial de e x e (equivale a una suma cuadrática de elementos)

def grad_w(x,y,w,b):
    return error(x,y,w,b)@x     #Sumatoria de todas las diferencias x la entrada

def grad_b(x,y,w,b):
    return np.sum(error(x,y,w,b))

def fit(x,y, lr=0.1,eps=1e-6):
    w = random.random()
    b = random.random()
    w_prev = -100
    b_prev = -100
    while(np.abs(w-w_prev)>eps or np.abs(b-b_prev)>eps):
        b_prev = b
        w_prev = w
        b=b-lr*grad_b(x,y,w,b)  #a medida que se aproxima al valor esperado, se reducen las variaciones sobre de w y b
        w=w-lr*grad_w(x,y,w,b)
        #e = error(x,y,w)
        print(f'w:{w}, eps:{w-w_prev}, loss:{loss(x,y,w,b)}, b:{b}') #'' para string, {} <- expresiones matematicas
    return w,b

fit(x,y)