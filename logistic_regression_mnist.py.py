import numpy as np
from sklearn import datasets
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

y =np.expand_dims(y, axis=1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y)
y= enc.transform(y).toarray()

# 데이터 섞어주기 위해 셔플

tmp = [[a,b] for a, b in zip(X, y)]
import random

random.shuffle(tmp)

X = [n[0] for n in tmp]
y = [n[1] for n in tmp]

X = np.asarray(X)
y = np.asarray(y)


d_size = 70000 


# 학습용데이터
train_x = X[0:int(d_size*0.85)]
train_y = y[0:int(d_size*0.85)]

# dev 데이터
dev_x = X[int(d_size*0.85):int(d_size*0.9)]
dev_y = y[int(d_size*0.85):int(d_size*0.9)]

# test 데이터
test_x = X[int(d_size*0.9):d_size]
test_y = y[int(d_size*0.9):d_size]

#파라미터 설정

epochs = 100 #최대에폭
learning_rate = 0.01
batch_size = 2000
dimension = 784
max_data = train_x.shape[0]

W = np.random.normal(loc=0.0, 
                        scale = np.sqrt(2/(dimension+10)), 
                        size = (dimension,10))
b = 0

train_batch = get_mini_batches(train_x, train_y, batch_size)[0] 

train_batch_x= train_batch[0]
train_batch_y= train_batch[1]

print(train_batch_x.shape)
print(train_batch_y.shape)

print(train_batch_y[42])
plt.imshow(train_batch_x[42].reshape((28, 28)), cmap='gray')


def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition



# train, test

dev_temp = 0

for i in range(epochs):
    for j in range(int(max_data/batch_size)):   
        
        gradient_w =  np.dot(np.transpose(train_batch_x), (softmax(np.dot(train_batch_x,W) + b) - train_batch_y ))/d_size
        gradient_b =  np.sum((np.dot(train_batch_x,W) + b - train_batch_y )* 2) /d_size

        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b

    train_hypothesis = softmax(np.dot(train_x,W) + b)
    train_est = np.argmax(train_hypothesis, axis=1) - np.argmax(train_y, axis=1)
    train_Loss = np.sum((train_hypothesis - train_y) ** 2) / d_size*0.1
    train_acc= len(np.where(train_est==0)[0])/int(d_size*0.85)*100
    
    dev_hypothesis = softmax(np.dot(dev_x,W) + b)
    dev_est = np.argmax(dev_hypothesis, axis=1) - np.argmax(dev_y, axis=1)
    dev_Loss = np.sum((dev_hypothesis - dev_y) ** 2) / d_size*0.1
    dev_acc = len(np.where(dev_est==0)[0])/int(d_size*0.05)*100
    
    test_hypothesis = softmax(np.dot(test_x,W) + b)
    test_est = np.argmax(test_hypothesis, axis=1) - np.argmax(test_y, axis=1)
    test_Loss = np.sum((test_hypothesis - test_y) ** 2) / d_size*0.1
    test_acc = len(np.where(test_est==0)[0])/int(d_size*0.1)*100
    
    
    print('Epoch  {:d}'.format(i+1))
    print('[Train]  Loss : {:f}    Acc :  {:.4f}'.format(np.sum(train_Loss), train_acc))
    print('[Dev]  Loss : {:f}    Acc :  {:.4f}'.format(np.sum(dev_Loss), dev_acc))
    print('[Test]  Loss : {:f}    Acc :  {:.4f}'.format(np.sum(test_Loss), test_acc))
    print()

    if(abs(dev_acc-dev_temp)<0.0001):
        print("Early Stopping !")
        break
    
    if((i+1)%5==0):
        dev_temp = dev_acc