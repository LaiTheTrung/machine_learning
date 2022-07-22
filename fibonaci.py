def fibonacci(n):
    if n==1:
        return 1
    elif n==2:
        return 2
    else:
        return fibonacci(n-1)+fibonacci(n-2)
# print(fibonacci(4))
import timeit
import numpy as np
def iter_fib(n):
    fib=np.ones(n)
    fib[1]=2
    for i in range(2,n,1):
        fib[i]=fib[i-1]+fib[i-2]
    return fib[n-1]
mysetup ='import numpy as np'
print (timeit.timeit(setup = mysetup,
                     stmt = lambda: iter_fib(13),
                     number = 10000))
print (timeit.timeit(stmt = lambda: fibonacci(13),
                     number = 10000))
