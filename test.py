from subprocess import list2cmdline
import numpy as np
x = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])

y= np.array([[1],[2]])
print(x**2)
values = np.unique(x)
list_Y=[]
def convert_Y(Ys,i):
        Ys[Ys!=i]=0
        Ys[Ys==i]=1
        return Ys
for value in values:
            print(value)          
            print(sample_Y)
            list_Y.append(sample_Y)
