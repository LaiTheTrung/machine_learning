
import numpy as np
import pandas as pd
import matplotlib as plt

from Classification_library import BinaryClassification,OneVsAllClassification

df=pd.read_csv("winequality-red.csv")
df=df.sample(frac=1)

Values=df["quality"].value_counts()


# DATA CLASSIFICATION
#############
Train1=df.loc[df["quality"].isin([5,6])].reset_index(drop=True)

Xs_df=Train1.drop(columns="quality").transpose()
n,m=Xs_df.shape
Ys_df=Train1[["quality"]].transpose()

LenThreshold=int(0.8*len(Train1))
Xs_train1 = Xs_df.iloc[:,:LenThreshold].to_numpy()
Ys_train1 = Ys_df.iloc[:,:LenThreshold].to_numpy()-5
Xs_test1 = Xs_df.iloc[:,LenThreshold:].to_numpy()
Ys_test1 = Ys_df.iloc[:,LenThreshold:].to_numpy()-5

#DATA ONE VS ALL
#############
Train2_4=df.loc[df["quality"]==4].head(53)
Train2_5=df.loc[df["quality"]==5].head(53)
Train2_6=df.loc[df["quality"]==6].head(53)
Train2_7=df.loc[df["quality"]==7].head(53)
Train2= pd.concat([Train2_4,Train2_5,Train2_6,Train2_7])
Train2 = Train2.reset_index(drop=True).sample(frac=1)

Xs_df=Train2.drop(columns="quality").transpose()
n,m=Xs_df.shape
Ys_df=Train2[["quality"]].transpose()

LenThreshold=int(0.8*len(Train2))
Xs_train2 = Xs_df.iloc[:,:LenThreshold].to_numpy()
Ys_train2 = Ys_df.iloc[:,:LenThreshold].to_numpy()
Xs_test2 = Xs_df.iloc[:,LenThreshold:].to_numpy()
Ys_test2 = Ys_df.iloc[:,LenThreshold:].to_numpy()
Xs_sqrt= np.sqrt(Xs_train1)




lr=0.1
rl=1
classification=BinaryClassification(Xs_train1,Ys_train1,lr,rl)

for epoch in range(1000):
    if epoch % 100==0:
        
        loss = classification.CostFunc()
        accuracy = classification.evaluate(Xs_test1,Ys_test1)
        print(f"loss epoch {epoch}: {loss}")       
        print(f"Accuracy epoch {epoch}: {accuracy}")
        print()
    classification.reAssign()

# lr=0.01
# rl=0.5
# classification=OneVsAllClassification(Xs_train2,Ys_train2,lr)
# for epoch in range(1000):
#     if epoch % 300==0:
        
#         loss = classification.TotalCostFunc()
#         TotalLoss=sum(loss)
#         accuracies= classification.evaluate(Xs_test2,Ys_test2)
#         print(f"loss epoch {epoch}: {loss}")
#         print(f"total loss epoch {epoch}: {TotalLoss}")       
#         print(f"Accuracy epoch {epoch}: {accuracies}")
#         print()
#     classification.TotalReAssign()