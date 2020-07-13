# pseudo inverse 이용

A = np.arange(10).reshape((2,5))
b = np.arange(6).reshape((2,3))

print(np.dot(np.linalg.pinv(A),b))