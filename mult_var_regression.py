import numpy as np
import pandas as pd






class mult_var_regression():
    def __init__(self,Xs,Ys,lr):
    
        self.Xs=self.makeup(Xs)
        self.Ys=Ys
        self.lr=lr
        self.n,self.m= self.Xs.shape
        self.theta=np.zeros((self.n,1))
        self.theta=np.random.uniform(low=-0.3, high = 0.3, size=(self.n,1))
    def makeup(self,Xs):
        Xs=np.transpose(Xs)
        Xs=self.Scaling(Xs)
        n,m=Xs.shape 
        added_X0=np.ones((1,m))
        add_sqrt=np.float_power(Xs,0.5) # sqrt function
        Xs = np.concatenate((added_X0,Xs),axis=0) #(X0,X1,X2,X3...Xn) matrix nxm with new n depending on how much variables you want to add
        Xs = np.concatenate((Xs,add_sqrt),axis=0)
        return Xs
    def Scailing(self,Xs):
        n,m=Xs.shape()
        mean_value=np.mean(Xs,axis=1).reshape(n,1)
        max_value=np.amax(Xs,axis=1).reshape(n,1)
        min_value=np.amin(Xs,axis=1).reshape(n,1)
        range=max_value-min_value
        return (Xs-mean_value)/range

    def hypothesis(self):
        hx_matrix= np.matmul(np.transpose(self.theta),self.Xs) # (1xn)*(nxm)=(1xm)
        return hx_matrix
        
    def CostFunction(self):
        J_matrix=(self.hypothesis()-self.Ys)**2 #1xm
        J=np.mean(J_matrix)*(1/2)
        return J

    def reAssigned(self):
        hx=self.hypothesis()
        dtheta= self.lr*(np.mean((hx-self.Ys)*self.Xs,axis=1).reshape(self.n,1)) # nx1
        self.theta=self.theta-dtheta

    def Regression(self):
        for epoch in range(10):
            J= self.CostFunction()
            print(J)
            self.reAssigned()

df=pd.read_csv("Fish.csv")
Xs=df[["Length1","Length2","Length3","Height","Width"]].to_numpy() #(mxn)
print(Xs.shape)
Ys=df["Weight"].to_list()
Ys=np.array([Ys]) #(1xm)       
M_LR=mult_var_regression(Xs,Ys,0.0001)
M_LR.Regression()
print(M_LR.theta)

