from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


style.use('fivethirtyeight')

xs= np.array([1,2,3,4,5])
ys= np.array([2,3,5,7,9,])

def bestslopeintercept(xs,ys):
    m=((mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs)))
    
    b= mean(ys)- m*mean(xs)
    
    return m,b


m,b= bestslopeintercept(xs,ys)

regressionline=[(m*x)+b for x in xs]

plt.scatter(xs,ys)
plt.plot(xs, regressionline)
plt.show()
