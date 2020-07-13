import numpy as np
from sklearn import datasets
from collections import OrderedDict

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



def normalize(x):
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_x = normalize(train_x)
dev_x = normalize(dev_x)
test_x = normalize(test_x)



class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
    
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t): # negative log likelihood
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class MLP:
    def __init__(self, layers_size, weight_init_std = 0.01):
        
        temp = len(layers_size)-2 # 히든 레이어 갯수
        self.temp = temp
        # 가중치 초기화
        self.params = {}
        
        self.params['W1'] = weight_init_std * np.random.randn(layers_size[0], layers_size[1])
        self.params['b1'] = np.zeros(layers_size[1])
        
        for i in range(temp):
            self.params['W'+str(i+2)] = weight_init_std * np.random.randn(layers_size[i+1], layers_size[i+2])
            self.params['b'+str(i+2)] = np.zeros(layers_size[i+2])
            

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        for i in range(temp):
            self.layers['Relu'+str(i+1)] = Relu()
            self.layers['Affine'+str(i+2)] = Affine(self.params['W'+str(i+2)], self.params['b'+str(i+2)])
           
        self.lastLayer = SoftmaxWithLoss()

    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers: #역전파
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        
        for i in range(self.temp):
            grads['W'+str(i+2)], grads['b'+str(i+2)] = self.layers['Affine'+str(i+2)].dW, self.layers['Affine'+str(i+2)].db
        return grads


# 파라미터 설정

#network = MLP([784,50,10]) # input(784고정) - [hidden] - output(10고정: 10개로 분류)
#network = MLP([784,50,50,10])
network = MLP([784,50,50,50,10])
#network = MLP([784,50,50,50,50,10])

iters_num = 10000000
train_size = train_x.shape[0]
batch_size = 500
learning_rate = 0.5
iter_per_epoch = max(train_size / batch_size, 1)
max_epoch = 100 # default = 100



wb_arrange = ['W1', 'b1']
tem=1
dev_temp=0
for i in range(network.temp):
    wb_arrange.append('W'+str(i+2))
    wb_arrange.append('b'+str(i+2))
    
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_x[batch_mask]
    t_batch = train_y[batch_mask]
    
    # backpropa
    grad = network.gradient(x_batch, t_batch)
    
    # update
    for key in wb_arrange:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    
    
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(train_x, train_y)
        dev_acc = network.accuracy(dev_x, dev_y)
        test_acc = network.accuracy(test_x, test_y)
        
        train_loss = network.loss(train_x, train_y)
        dev_loss = network.loss(dev_x, dev_y)
        test_loss = network.loss(test_x, test_y)
        print('Epoch  {:d}'.format(tem))
        print('[Train]  Loss : {:f}    Acc :  {:.4f}'.format(train_loss, train_acc*100))
        print('[Dev]  Loss : {:f}    Acc :  {:.4f}'.format(dev_loss, dev_acc*100))
        print('[Test]  Loss : {:f}    Acc :  {:.4f}'.format(test_loss, test_acc*100))
        print()
        
        if tem==max_epoch:
            break        
        tem=tem+1
        
        
        if(abs(dev_acc*100-dev_temp)<0.2):
            print("Early Stopping !")
            break

        dev_temp = dev_acc*100