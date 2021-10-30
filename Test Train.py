import numpy as np




# add 1s column
x = np.c_[np.ones([n, 1]), x]

n = len(x)
_lambda = 4


n, d = x.shape
d = d-1  # remove 1 for the extra column of ones we added to get the original num features

# construct reg matrix
reg_matrix = _lambda * np.eye(d + 1)
reg_matrix[0, 0] = 0

# analytical solution (X'X + regMatrix)^-1 X' y
np.linalg.pinv(x.T.dot(x) + reg_matrix).dot(x.T).dot(y)