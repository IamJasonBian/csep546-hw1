import numpy as np

arr =  np.array([2,3,1,0])
num_classes = 4
        
init_flag = 0


for i in arr:
    row = np.zeros(num_classes, dtype=int)
    row[int(i)] = 1

    if init_flag == 0:
        base = row
        init_flag = 1
    else:
        base = np.vstack((base, row))
        
base
