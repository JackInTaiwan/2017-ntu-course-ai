from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
a = [i for i in range(1010)]
b = [0 for i in range(1000)]
for i in range(10) :
    b.append(1)
a = np.array(a)
b = np.array(b)
x_1, x_2, y_1, y_2 = train_test_split(a, b, test_size=0.3)
print (y_2)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
for train_index, test_index in sss.split(a, b) :
    print (test_index)
    y_test = b[test_index]
print (y_test)