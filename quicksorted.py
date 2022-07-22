def quickSort(lst):
    if len(lst)==0:
        return lst 
    elif len(lst)>=1:
        n=lst[0]
        bigger=[]
        smaller=[]
        same=[]
        for i in lst:
            if i>n: bigger.append(i)
            elif i<n: smaller.append(i)
            else: same.append(i)
        return quickSort(smaller)+same+quickSort(bigger)
lst= [12,34,3,4,5,6,3,43,5,56,23,45,6,44,3,45,6,46,3,2,321,12]
print(quickSort(lst))

print( )