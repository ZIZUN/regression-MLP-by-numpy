import numpy as np
from matplotlib import pyplot as plt
import pickle

## 데이터셋 만들고 저장

w = 20*np.random.rand(1,100)-10 
b = 20*np.random.rand(1)-10

x = 20*np.random.rand(10000,100,1)-10
y = np.matmul(w,x)

x=np.squeeze(x, axis=2)
y=np.squeeze(y, axis=2)+b

print(w.shape)
print(b.shape)

print(x.shape)
print(y.shape)



# pickle 이용해 저장

with open("linear/myrandomdataset_w.pickle","wb") as fw:
    pickle.dump(w, fw)
with open("linear/myrandomdataset_b.pickle","wb") as fw:
    pickle.dump(b, fw)
with open("linear/myrandomdataset_x.pickle","wb") as fw:
    pickle.dump(x, fw)
with open("linear/myrandomdataset_y.pickle","wb") as fw:
    pickle.dump(y, fw)

print(y.shape)
plt.figure
plt.hist(y, 444)
plt.show()