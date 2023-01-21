import numpy as np
from math import exp
def func(x):
    f=2*exp(x[0])*x[1]+3*x[0]*x[1]**2
    return f
def eh(x,i):
    delx=np.repeat(0.001,x.size)
    eh=np.zeros(x.size)
    eh[i]=1
    return eh*delx
def compute_gradient(f,x):
    h=0.001
    g=np.zeros(x.size)
    for i in range(x.size):
        g[i]=(f(x+eh(x,i))-f(x-eh(x,i)))/(2*h)
    return g
def compute_hessian(f,x):
    delx=np.repeat(0.001,x.size)
    hs=np.zeros([x.size,x.size])
    for i in range(x.size):
        for j in range(x.size):
            if(i==j):
                hs[i][j]=(1/(delx[i])**2)*(f(x+eh(x,i))-2*f(x)+f(x-eh(x,i)))
            else:
                A=f(x+eh(x,i)+eh(x,j))
                B=f(x-eh(x,i)-eh(x,j))
                C=f(x-eh(x,i)+eh(x,j))
                D=f(x+eh(x,i)-eh(x,j))
                hs[i][j]=(1/(4*delx[i]*delx[j]))*(A+B-C-D)
    return hs   
x=np.array([1,1]) 
print(f"f(x) = \n {func(x)}")
print(f"gradient(f(x),x) = \n{compute_gradient(func,x)}")
print(f"hessian(f(x),x) = \n {compute_hessian(func,x)}")