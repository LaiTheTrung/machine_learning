import random
import numpy as np
import matplotlib.pyplot as plt
arr= arr = np.array([[1.20, 2.20, 3.00], [4.14, 5.65, 6.42]])
np.savetxt("multivariate.txt", arr, fmt="%.2f",
header = "Col1 Col2 Col3")
data = np.loadtxt("multivariate.txt")
Xs = data[:,0]
Ys = data[:,1]

plt.plot(Xs,Ys,"rx")
#plt.plot(pointx,pointy)
plt.xlabel("Price")
plt.ylabel("Size")
#plt.show()

def hypothesis(theta0,theta1,x):
    hx = theta0 + theta1*x
    return hx

def J(theta0,theta1,Xs,Ys):
    totalJ = 0
    for i in range(len(Xs)):
        totalJ += (hypothesis(theta0,theta1,Xs[i]) - Ys[i])**2
    finalJ = totalJ/(2*len(Xs))
    return finalJ

def reAssigned(theta0,theta1,Xs,Ys,lr):
    dJ0 = 0
    dJ1 = 0
    m = len(Xs)

    for i in range(len(Xs)):
        hx = hypothesis(theta0,theta1,Xs[i])
        dj0 = hx - Ys[i]
        dj1 = (hx - Ys[i])*Xs[i]
        dJ0 += dj0
        dJ1 += dj1

    theta0 = theta0 - lr*dJ0/m
    theta1 = theta1 - lr*dJ1/m

    return theta0,theta1

def visualize(Xs,Ys,theta0,theta1):
  xmin = min(Xs)
  xmax = max(Xs)
  # ymin = min(Ys)
  # ymax = max(Ys)
  yhmin = hypothesis(theta0,theta1,xmin)
  yhmax = hypothesis(theta0,theta1,xmax)

  pointx = [xmin,xmax]
  pointy = [yhmin,yhmax]

  plt.plot(Xs,Ys,"rx")
  plt.plot(pointx,pointy)
  plt.xlabel("Price")
  plt.ylabel("Size")
  plt.show()

theta0 = random.randint(-3,4)
theta1 = random.randint(-3,4)
epoch = 15
lr = 0.0000003

for i in range(epoch):
    loss = J(theta0,theta1,Xs,Ys)
    print(f"Epoch {i+1}: {loss}")
    #visualize(Xs,Ys,theta0,theta1)
    theta0,theta1 = reAssigned(theta0,theta1,Xs,Ys,lr)