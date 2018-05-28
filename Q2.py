


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import random
import pickle
import gzip

import time
import datetime
import os

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import issparse
from scipy.sparse import vstack



def standardization(xx):
    n,m = xx.shape
    eps = 1e-5
    std = np.std(xx,axis=0)
    u_j = np.mean(xx,axis=0)
    return (xx - u_j)/(std+eps)


def adjust_lr(optimizer, lr0, epoch, total_epochs):
    lr = lr0 * (0.36 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def prediction(data_batch,model):
    correct = 0
    total = 0
    for x, y in data_batch:
        x,y = model.input(x,y)
        out = model.forward(x)
        _, pred = torch.max(out.data, 1)
        pred = Variable(pred)
        total += float(y.size(0))
        
        correct += float((pred == y).sum())
    return 100*correct/total


def linear_ini(LL,initialization):
    '''
    inputs : linear layer (LL) and the initialization
    output : linear layer with the chosen initialization
    '''
    if initialization == 'zero':
        LL.weight.data = nn.init.constant(LL.weight.data, 0)
        LL.bias.data = nn.init.constant(LL.bias.data, 0)
    
    if initialization == 'normal':
        LL.weight.data = nn.init.normal(LL.weight.data, 0,1)
        LL.bias.data = nn.init.normal(LL.bias.data, 0,1)

    if initialization == 'glorot':
        LL.weight.data = nn.init.xavier_uniform(LL.weight.data, gain=1)
        LL.bias.data = nn.init.constant(LL.bias.data, 0)
    if initialization == 'default': 
        pass
        
    return LL


class MLPLinear(nn.Module):
	def __init__(self, dimensions, cuda):
		super(MLPLinear, self).__init__()
		self.h0 = dimensions[0]
		self.h1 = dimensions[1]
		self.h2 = dimensions[2]        

		self.fc1 = torch.nn.Linear(self.h0,self.h1)
		self.fc2 = torch.nn.Linear(self.h1,self.h2)        
		self.relu = nn.ReLU()
		self.criterion = nn.CrossEntropyLoss()
		self.cuda = cuda

		if cuda: 
			self.fc1.cuda()
			self.fc2.cuda()
			self.relu.cuda()
			self.criterion.cuda()

	def initialization(self,method):
		self.fc1 = linear_ini(self.fc1,method)
		self.fc2 = linear_ini(self.fc2,method)

	def input(self,x,y):
		if self.cuda : 
			x = Variable(x.cuda())
			y = Variable(y.cuda())
		else: 
			x = Variable(x)
			y = Variable(y)
		return x,y
        
    
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return  out




######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################


path1 = "./20news-bydate/"
vocabulary = np.loadtxt(os.path.join(path1, "vocabulary.txt"), dtype = str)

path2 = path1 + "matlab/"

train_valid_x_ = np.loadtxt(os.path.join(path2, "train.data"), dtype = np.uint16)
train_valid_y_ = np.loadtxt(os.path.join(path2, "train.label"), dtype = np.uint8)
test_x_ = np.loadtxt(os.path.join(path2, "test.data"), dtype = np.uint16)
test_y_ = np.loadtxt(os.path.join(path2, "test.label"), dtype = np.uint8)

n_words_vocabulary = len(vocabulary)
n_doc_train_data = int(train_valid_x_[-1,0])
n_doc_test_data = int(test_x_[-1,0])

train_valid_x = np.empty((n_doc_train_data, n_words_vocabulary))
test_x = np.empty((n_doc_test_data, n_words_vocabulary))


### Filling of train_x and test_x from train_data and test_data
for i in range(train_valid_x_.shape[0]):
    j,k,l = train_valid_x_[i]
    train_valid_x[j-1,k-1]=l

for i in range(test_x_.shape[0]):
    j,k,l = test_x_[i]
    test_x[j-1, k-1]=l


documents_x = vstack([train_valid_x, test_x])
inv_idf = ((documents_x != 0).sum(axis = 0))
idf = np.log(documents_x.shape[0]/inv_idf)

train_valid_x_idf = np.multiply(train_valid_x,idf)
test_x_idf = np.multiply(test_x,idf)
train_valid_x_std = standardization(train_valid_x)
test_x_std = standardization(test_x)

train_valid_x = torch.FloatTensor(train_valid_x)
train_valid_x_idf = torch.FloatTensor(train_valid_x_idf)
train_valid_x_std = torch.FloatTensor(train_valid_x_std)
train_valid_labels = torch.FloatTensor(train_valid_y_-1)

test_data = torch.FloatTensor(test_x)
test_data_idf = torch.FloatTensor(test_x_idf)
test_data_std = torch.FloatTensor(test_x_std)
test_labels = torch.FloatTensor(test_y_-1)


torch.manual_seed(1000)

ind = torch.randperm(train_valid_x.shape[0])
train_valid_x = train_valid_x[ind]
train_valid_x_idf = train_valid_x_idf[ind]
train_valid_x_std = train_valid_x_std[ind]
train_valid_labels = train_valid_labels[ind]


train_x = train_valid_x[:9015]
train_x_idf = train_valid_x_idf[:9015]
train_x_std = train_valid_x_std[:9015]
train_y = train_valid_labels[:9015].long()

valid_x = train_valid_x[9015:]
valid_x_idf = train_valid_x_idf[9015:]
valid_x_std = train_valid_x_std[9015:]
valid_y = train_valid_labels[9015:].long()

test_x = test_data
test_x_idf = test_data_idf
test_x_std = test_data_std
test_y = test_labels.long()

print('Data Loaded')


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

# this is a test - for Iban. 
batch_size = 20
num_epochs = 20
lr0 = 0.01

h0 = 61188    #784
h1 = b1 = 100
h2 = b2 = 20
cuda = True

xx = torch.Tensor(np.empty((9015,61188)))
yy = torch.Tensor(np.empty((9015))).long()

model = MLPLinear([h0,h1,h2],cuda) 
model.initialization('glorot')
xt,yt = model.input(xx,yy)
pred_batch = model.forward(xt)
loss_batch = model.criterion(pred_batch,yt)
print('sucess!')

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

''' hunt for HP '''
# np.random.seed(seed=None)

# cuda = True
# h0 = 61188
# h1 = b1 = 100
# h2 = b2 = 20
# num_epochs = 20

# initialization_method = ['no_processing','tf_idf','standardization']
# batch_size = [133,133,133]
# lr0 = [0.0078,0.0078,0.0078]

# nb_hp = 20

# hyperparameters=np.empty((nb_hp,2))
# for i in range(nb_hp):
#     hyperparameters[i,:] = np.concatenate((np.random.randint(20,300,1), np.random.uniform(0.0001,0.01,1))).reshape(1,2)

# hp=[]
# for i,h in enumerate(hyperparameters):
#     hp.append([h[0],h[1],'no_processing'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],'tf_idf'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],'standardization'])    

# ''' LR ...  '''

# np.random.seed(seed=None)

# cuda = True
# h0 = 61188
# h1 = b1 = 100
# h2 = b2 = 20
# num_epochs = 20

# initialization_method = ['no_processing','tf_idf','standardization']
# batch_size = [133,133,133]
# lr0 = [0.0078,0.0078,0.0078]

# nb_hp = 15

# hyperparameters=np.empty((nb_hp,2))
# for i in range(nb_hp):
#     hyperparameters[i,:] = np.concatenate((np.array((150, )), np.random.uniform(0.0001,0.1,1))).reshape(1,2)
    
# #np.random.randint(20,300,1)

# # learning rate
# hyperparameters = np.hstack((150* np.ones(12).reshape(12,1),np.logspace(-5.,-1, num=12).reshape(12,1)))

# hp = []
# for i,h in enumerate(hyperparameters):
#     hp.append([h[0],h[1],'no_processing'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],'tf_idf'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],'standardization'])    
# print(len(hp))



''' standardization -- epsilon = 0 '''

np.random.seed(seed=None)

cuda = True
h0 = 61188
h1 = b1 = 100
h2 = b2 = 20
num_epochs = 20

initialization_method = ['standardization']
batch_size = [133,133,133]
lr0 = [0.0078,0.0078,0.0078]

nb_hp = 10

hyperparameters=np.empty((nb_hp,2))
for i in range(nb_hp):
    hyperparameters[i,:] = np.concatenate((np.array((150, )), np.random.uniform(0.0001,0.1,1))).reshape(1,2)
    
#np.random.randint(20,300,1)

# learning rate
# hyperparameters = np.hstack((150* np.ones(12).reshape(12,1),np.logspace(-5.,-1, num=12).reshape(12,1)))

hp = []
# for i,h in enumerate(hyperparameters):
#     hp.append([h[0],h[1],'no_processing'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],'tf_idf'])
for i,h in enumerate(hyperparameters):    
    hp.append([h[0],h[1],'standardization'])    
print(len(hp))



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
n = len(hp)
m = num_epochs
loss_train = np.empty([n,m]) 
loss_valid = np.empty([n,m]) 
acc_train = np.empty([n,m]) 
acc_valid = np.empty([n,m]) 
lr = np.empty([n,m]) 

bs = batch_size[0]
lrs = lr0[0]

for i, (bs, lrs, method_) in enumerate((hp)):
    t0 = time.time()
    print('Iteration: '+str(i)+'/'+str(n))
    model = MLPLinear([h0,h1,h2],cuda)
    model.initialization('glorot')
    optimizer = optim.SGD(model.parameters(),lr=lrs,momentum=0.9)
    if method_ == 'no_processing':
        xt = train_x
        yt = train_y
        xv = valid_x
        yv = valid_y

    elif method_ == 'tf_idf':
        xt = train_x_idf
        yt = train_y
        xv = valid_x_idf
        yv = valid_y

    elif method_ == 'standardization':
        xt = train_x_std
        yt = train_y
        xv = valid_x_std
        yv = valid_y
  
    else: print('You made a mistake, pal! ')
    
    train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(xt,yt), batch_size=bs, shuffle=True)
    valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(xv,yv), batch_size=bs, shuffle=False)

    print('     ')
    print('     ')    
    print('Preprocessing procedures : '+method_+'   bs ='+str(bs)+'   lrs ='+str(lrs))
    print('________________________________')    
    for j in range(num_epochs):
        model_loss = 0
        for batch_idx, (xx,yy) in enumerate(train_batch):
            xx,yy = model.input(xx,yy)
            pred_batch = model.forward(xx)

            optimizer.zero_grad()
            loss_batch = model.criterion(pred_batch,yy)
            loss_batch.backward()
            optimizer.step()  

    
        xt_,yt_ = model.input(xt,yt)
        xv_,yv_ = model.input(xv,yv)
        pred_train_all = model.forward(xt_)
        pred_valid_all = model.forward(xv_)  
  
        loss_train[i,j] = model.criterion(pred_train_all,yt_).data[0]
        loss_valid[i,j] = model.criterion(pred_valid_all,yv_).data[0]


        acc_train[i,j]  = prediction(train_batch,model)
        acc_valid[i,j]  = prediction(valid_batch,model)
            
        lr_ = adjust_lr(optimizer,lrs, j+1, num_epochs)
        if j%5 == 0: print('Epoch #'+str(j)+', Train loss = '+str(loss_train[i,j])+', Valid loss = '+str(loss_valid[i,j]))
    ts = time.time()
    print('Time = %.2f sec'%(ts - t0))
print('done!')

filename = 'epsilon'
with open(filename, 'wb') as f: 
    pickle.dump([loss_train, loss_valid, acc_train, acc_valid, hp], f)
print(filename)




