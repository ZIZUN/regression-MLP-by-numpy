from numpy.linalg import inv
# Ax = b 풀기

A = np.arange(4).reshape((2,2))
b = np.arange(6).reshape((2,3))

print(np.dot(np.linalg.inv(A),b))

