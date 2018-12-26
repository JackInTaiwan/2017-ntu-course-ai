import numpy as np



a = [[1,2,3],[2,3,4]]
a = np.array(a)

for row in a :
    if row[0] == 1 : row[0] = 9

print (a)
