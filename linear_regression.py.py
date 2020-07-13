import numpy as np
from matplotlib import pyplot as plt
import pickle

from numpy import random

#배치 뽑는 함수

def get_mini_batches(X, y, batch_size): 
    random_idxs = np.random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches

d_size = 1000 #deafult   N=1000
if(x_.shape[0]==10000): # N의 사이즈에 따른 설정 (1000, 10000, 100000 중 택 1)
    d_size=10000
elif(x_.shape[0]==100000):
    d_size=100000
elif(x_.shape[0]==442):
    d_size=442
    
# 학습용데이터
train_x = x_[0:int(d_size*0.85)]
train_y = y_[0:int(d_size*0.85)]

# dev 데이터
dev_x = x_[int(d_size*0.85):int(d_size*0.9)]
dev_y = y_[int(d_size*0.85):int(d_size*0.9)]

# test 데이터
test_x = x_[int(d_size*0.9):d_size]
test_y = y_[int(d_size*0.9):d_size]

print(test_x.shape)
print(test_y.shape)

#파라미터 설정

epochs = 100 # 최대 에폭
learning_rate = 0.01
batch_size = 300
dimension = 100
max_data = train_x.shape[0]

W = np.zeros((dimension,1))
b = 0

train_batch = get_mini_batches(train_x, train_y, batch_size)[0] 

train_batch_x= train_batch[0]
train_batch_y= train_batch[1]

print(train_batch_x.shape)
print(train_batch_y.shape)


# train, test

dev_temp = 0

for i in range(epochs):
    for j in range(int(max_data/batch_size)):    
        gradient_w =  np.dot(np.transpose(train_batch_x), (np.dot(train_batch_x,W) + b - train_batch_y ))* 2 / batch_size
        gradient_b =  np.sum((np.dot(train_batch_x,W) + b - train_batch_y )* 2) / batch_size

        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
    
        hypothesis = np.dot(train_batch_x,W) + b
        Loss = np.sum((hypothesis - train_batch_y) ** 2) / batch_size

    train_hypothesis = np.dot(train_x,W) + b
    train_Loss = np.sum((train_hypothesis - train_y) ** 2) / d_size*0.1
        
    dev_hypothesis = np.dot(dev_x,W) + b
    dev_Loss = np.sum((dev_hypothesis - dev_y) ** 2) / d_size*0.1
    
    test_hypothesis = np.dot(test_x,W) + b
    test_Loss = np.sum((test_hypothesis - test_y) ** 2) / d_size*0.1
    
    w_squared_error = np.sum((W-np.transpose(w_))** 2) / dimension
    b_squared_error = np.sum(abs(b-b_))
    
    print('Epoch  {:d}\n[Train Loss]  {:f}'.format(i+1, train_Loss))
    print('[Dev Loss]  {:f}'.format(np.sum(dev_Loss)))
    print('[Test Loss]  {:f}'.format(np.sum(test_Loss)))
    print('w squared error : {:f} , b squared error : {:f}'.format(w_squared_error, b_squared_error))
    print()

    if(dev_temp != 0 and (dev_temp - dev_Loss<0 or dev_temp - dev_Loss<0.00000001)):
        print("Early Stopping !")
        break

    dev_temp = test_Loss