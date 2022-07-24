import numpy as np
import pandas as pd
from Neural_Network import NeuralNetwork

df=pd.read_csv("winequality-red.csv")
df=df.sample(frac=1)


########################### mulitble 
Train_4= df.loc[df["quality"]==4]
Train_5= df.loc[df["quality"]==5].head(53)
Train_6= df.loc[df["quality"]==6].head(53)
Train_7= df.loc[df["quality"]==7].head(53)
Train=pd.concat([Train_4,Train_5,Train_6,Train_7]).sample(frac=1)

Xs=Train[["fixed acidity","volatile acidity","citric acid","residual sugar"]].to_numpy().transpose()
n,m=Xs.shape
Ys=Train[["quality"]].to_numpy().reshape(1,m)
LenThreshold=int(0.8*m)
Xs_train=Xs[:,:LenThreshold]
Ys_train=Ys[:,:LenThreshold]
Xs_test=Xs[:,LenThreshold:]
Ys_test=Ys[:,LenThreshold:]

######################## binary
Train1=df.loc[df["quality"].isin([5,6])].reset_index(drop=True)

Xs_df=Train1.drop(columns="quality").transpose()
n,m=Xs_df.shape
Ys_df=Train1[["quality"]].transpose()

LenThreshold=int(0.8*len(Train1))
Xs_train1 = Xs_df.iloc[:,:LenThreshold].to_numpy()
Ys_train1 = Ys_df.iloc[:,:LenThreshold].to_numpy()-5
Xs_test1 = Xs_df.iloc[:,LenThreshold:].to_numpy()
Ys_test1 = Ys_df.iloc[:,LenThreshold:].to_numpy()-5

########################
#BINARY TRAINING SET
num_layers=4
num_unit_per_layer=5
lr=1
rl=1.5
BNW = NeuralNetwork(Xs_train1,Ys_train1,lr,rl,num_layers,num_unit_per_layer)
for epoch in range (1000):
   
    if epoch % 100==0:
        totalLoss=BNW.CostFunc()
        accuracy = BNW.evaluate(Xs_test1,Ys_test1)
        print ("TotalLoss: ",totalLoss)
        print ("accuracy",accuracy)
        print()
    BNW.BackPropagation()
    BNW.ReAssign()

# MULTIBLE 
# num_layers=4
# num_unit_per_layer=5
# lr=1
# rl=1
# BNW = NeuralNetwork(Xs_train,Ys_train,lr,rl)
# for epoch in range (1000):
   
#     if epoch % 100==0:
#         totalLoss=BNW.CostFunc()
#         accuracy = BNW.evaluate(Xs_test,Ys_test)
#         print ("TotalLoss: ",totalLoss)
#         print ("accuracy",accuracy)

#         print()
#     BNW.BackPropagation()
#     BNW.ReAssign()




