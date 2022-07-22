import random
import numpy as np
import matplotlib.pyplot as plt

def hypothesis(theta0,theta1,x):
    return theta0+theta1*x
class my_Linear_Regression():
   
    def __init__(self,features_X,target_Y,learning_rate):
        self.X=features_X
        self.Y=target_Y
        self.lr=learning_rate
        self.theta0=0
        self.theta1=0
        self.J=0
  
    def costFunction(self):
        self.J=0
        for x,y in zip(self.X,self.Y):
            hx=hypothesis(self.theta0,self.theta1,x)
            self.J += (1/(2*len(self.X)))*(hx-y)**2
        
    def ReAssign(self):
        dJ_0=0
        dJ_1=0
        for x,y in zip(self.X,self.Y):
                dJ_0 += (hypothesis(self.theta0,self.theta1,x)-y)
                dJ_1 += (hypothesis(self.theta0,self.theta1,x)-y)*x
        self.theta0=self.theta0-self.lr*dJ_0*(1/len(self.X))
        self.theta1=self.theta1-self.lr*dJ_1*(1/len(self.X))

    def Regression(self):
       
        self.theta0=random.randint(-100,100)
        self.theta1=random.randint(-100,100)
        
        for epoch in range(10):         
            self.costFunction()
            print(self.J)
            self.ReAssign()

    def visualize(self):
        xmin = min(self.X)
        xmax = max(self.X)
        hxMin=hypothesis(self.theta0,self.theta1,xmin)
        hxMax=hypothesis(self.theta0,self.theta1,xmax)
        x_values= [xmin,xmax]
        y_values= [hxMin,hxMax]
        plt.plot(x_values,y_values)
        plt.plot(self.X, self.Y,"rx")
        plt.show()

    def predict(self,feature_x):
        return hypothesis(self.theta0,self.theta1,feature_x)





lines = np.loadtxt("linear_regression_1_variable.txt", delimiter=",")

sizes,prices=[],[]
for line in lines :
    size,bedroom, price = line
    sizes.append(int(size)) 
    prices.append(int(price))
X=sizes
Y=prices

LR=my_Linear_Regression(X,Y,0.0000001)
LR.Regression()
print(LR.theta0,LR.theta1)
print(LR.predict(750))
