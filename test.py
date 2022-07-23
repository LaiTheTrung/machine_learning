from subprocess import list2cmdline
import numpy as np
a= np.random.uniform(low=0.1, high = 0.5, size=(10,1))
print(a.shape)
print(np.array([0]).shape)
print(np.concatenate([np.array([0]),a[1:]],axis=0))
