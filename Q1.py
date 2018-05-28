import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import random
import pickle
import gzip

import time
import datetime

from sklearn.utils.extmath import cartesian


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

import pickle

with open('mnist.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, valid, test = u.load()

train_x, train_y = train
valid_x, valid_y = valid
test_x, test_y = test

train_x = train_x.reshape(50000,28,28)
valid_x = valid_x.reshape(10000,28,28)
test_x = test_x.reshape(10000,28,28)

train_data = torch.FloatTensor(train_x)
train_data_ = Variable(train_data.view(-1,784))
train_labels = torch.FloatTensor(train_y).long()
train_labels_ = Variable(train_labels.view(-1))

valid_data = torch.FloatTensor(valid_x)
valid_data_ = Variable(valid_data.view(-1,784))
valid_labels = torch.FloatTensor(valid_y).long()
valid_labels_ = Variable(valid_labels.view(-1))

test_data = torch.FloatTensor(test_x)
test_data_ = Variable(valid_data.view(-1,784))
test_labels = torch.FloatTensor(test_y).long()
test_labels_ = Variable(valid_labels.view(-1))



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


def flatten(x):
    return x.view(x.size()[0], -1)
    
def adjust_lr(optimizer,lrs, epoch, total_epochs):
    lr = lrs * (0.36 ** (epoch / float(total_epochs)))
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


def batch_loss(data_batch,model,criterion): 
    loss = 0
    total = 0
    for batch_idx, (images,labels) in enumerate(data_batch):
#     for images, labels in data_batch:
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)  
        outputs = model(images)
        loss += criterion(outputs, labels)
        total += labels.size(0)
        
    return loss.data[0]/total


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
        # that is important, see paper. 
        LL.bias.data = nn.init.constant(LL.bias.data, 0)
    if initialization == 'default': 
        pass
    return LL


class MLPLinear(nn.Module):
	def __init__(self, dimensions, cuda):
		super(MLPLinear, self).__init__()
		self.h0 = int(dimensions[0])
		self.h1 = int(dimensions[1])
		self.h2 = int(dimensions[2])       
		self.h3 = int(dimensions[3])

		self.fc1 = torch.nn.Linear(self.h0,self.h1)
		self.fc2 = torch.nn.Linear(self.h1,self.h2)        
		self.fc3 = torch.nn.Linear(self.h2,self.h3)                
		self.relu = nn.ReLU()
		self.criterion = nn.CrossEntropyLoss()
		self.cuda = cuda

		if cuda: 
			self.fc1.cuda()
			self.fc2.cuda()
			self.fc3.cuda()            
			self.relu.cuda()
			self.criterion.cuda()

	def initialization(self,method):
		self.fc1 = linear_ini(self.fc1,method)
		self.fc2 = linear_ini(self.fc2,method)
		self.fc3 = linear_ini(self.fc3,method)        

	def input(self,x,y):
		x = (x.view(-1,784))
		y = (y)  
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
		out = self.relu(out)        
		out = self.fc3(out)        
		return  out
    




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


# ### TO COMPARE INITIALIZATION - FOR ONE SET OF HYPERPARAMTERS. 

# # important information 
# batch_size = 100
# num_epochs = 1
# lr0 = 0.1

# h0 = 28*28    #784
# h1 = b1 = 500
# h2 = b2 = 300
# h3 = b3 = 10

# print('Number of parameters = '+str((h0*h1 + h1*h2 + h2*h3 + b1 + b2 + b3)/1E6)+'M')


# cuda = False
# initialization_method = ['default']#,'zero','normal','glorot'

# n = len(initialization_method)
# m = num_epochs

# loss_train = np.empty([n,m]) 
# loss_valid = np.empty([n,m]) 

# acc_train = np.empty([n,m]) 
# acc_valid = np.empty([n,m]) 

# train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=batch_size, shuffle=True)
# valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
# test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)

# t0 = time.time()        
# for i, method in enumerate(initialization_method):
#     model = MLPLinear([h0,h1,h2,h3],cuda) 
#     model.initialization(method)
#     optimizer = optim.SGD(model.parameters(), lr=0.01)
    
#     print('     ')
#     print('     ')    
#     print('Initialization method : '+method)
#     print('________________________________')    
#     for j in range(num_epochs):
#         model_loss = 0
#         for batch_idx, (x,y) in enumerate(train_batch):
#             xt,yt = model.input(x,y)
#             pred_batch = model.forward(xt)
            
#             optimizer.zero_grad()
#             loss_batch = model.criterion(pred_batch, yt)
#             loss_batch.backward()
#             optimizer.step()

#         xt_,yt_ = model.input(train_data,train_labels)
#         xv_,yv_ = model.input(valid_data,valid_labels)
#         pred_train_all = model.forward(xt_)
#         pred_valid_all = model.forward(xv_)  
        
#         loss_train[i,j] = model.criterion(pred_train_all,yt_).data[0]
#         loss_valid[i,j] = model.criterion(pred_valid_all,yv_).data[0]

#         acc_train[i,j]  = prediction(train_batch,model)
#         acc_valid[i,j]  = prediction(valid_batch,model)
        
#         print('Epoch #'+str(j)+', Train loss = '+str(loss_train[i,j])+', Valid loss = '+str(loss_valid[i,j]))
        
#     print('Train accuracy = '+str(prediction(train_batch,model))+'%')
#     ts = time.time()        
#     print('Time : %.1f sec'%(ts-t0))
# print('done!')




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


# #### HUNT FOR THE BEST HYPERPARAMTERS 

# np.random.seed(seed=None)


# batch_size = 100
# num_epochs = 100
# # lr0 = 0.1

# h0 = 28*28    #784
# h1 = b1 = 500
# h2 = b2 = 300
# h3 = b3 = 10

# nb_hp = 15
# hyperparameters=np.empty((nb_hp,4))
# for i in range(nb_hp):
#     hyperparameters[i,:] = np.concatenate((
#         np.random.randint(300,800,1),
#         np.random.randint(300,800,1),
#         np.random.randint(20,200,1),        
#         np.random.uniform(0.0001,0.1,1)
#     )).reshape(1,4)

# # initialization_method = ['default','zero','normal','glorot']    

# hp = []
# for i,h in enumerate(hyperparameters):
#     hp.append([h[0],h[1],h[2],h[3],'default'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],h[2],h[3],'zero'])
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],h[2],h[3],'normal'])    
# for i,h in enumerate(hyperparameters):    
#     hp.append([h[0],h[1],h[2],h[3],'glorot'])        


# cuda = True
# n = len(hp)
# m = num_epochs

# loss_train = np.empty([n,m]) 
# loss_valid = np.empty([n,m]) 

# acc_train = np.empty([n,m]) 
# acc_valid = np.empty([n,m]) 
# lr = np.empty([n,m]) 

# t0 = time.time()        
# for i, (h1s, h2s, bs, lrs, method_) in enumerate(hp):
# 	print('Iteration : %.f / %.f'%(i, len(hp)))
# 	train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=bs, shuffle=True)
# 	valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=bs, shuffle=False)
# 	test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=bs, shuffle=False)

# 	model = MLPLinear([h0,h1s,h2s,h3],cuda) 
# 	model.initialization(method_)
# 	optimizer = optim.SGD(model.parameters(), lr=lrs)
    
# 	print('     ')
# 	print('     ')    
# 	print('Initialization method : '+method_)
# 	print('________________________________')    
# 	for j in range(num_epochs):
# 		for batch_idx, (x,y) in enumerate(train_batch):
# 			xt,yt = model.input(x,y)
# 			pred_batch = model.forward(xt)

# 			optimizer.zero_grad()
# 			loss_batch = model.criterion(pred_batch, yt)
# 			loss_batch.backward()
# 			optimizer.step()

# 		xt_,yt_ = model.input(train_data,train_labels)
# 		xv_,yv_ = model.input(valid_data,valid_labels)
# 		pred_train_all = model.forward(xt_)
# 		pred_valid_all = model.forward(xv_)  
        

# 		loss_train[i,j] = model.criterion(pred_train_all,yt_).data[0]
# 		loss_valid[i,j] = model.criterion(pred_valid_all,yv_).data[0]

# 		acc_train[i,j]  = prediction(train_batch,model)
# 		acc_valid[i,j]  = prediction(valid_batch,model)

# 		if j%10==0: print('Epoch #'+str(j)+', Train loss = '+str(loss_train[i,j])+', Valid loss = '+str(loss_valid[i,j]))

# 		lr[i,j] = adjust_lr(optimizer,lrs, j+1, num_epochs)

# 	print('Train accuracy = '+str(prediction(train_batch,model))+'%')
# 	ts = time.time()        
# 	print('Time : %.1f sec'%(ts-t0))
# print('done!')


# filename = 'Hunt for best hyperparameters - Q1'
# with open(filename, 'wb') as f: 
#     pickle.dump([loss_train, loss_valid, acc_train, acc_valid, hp], f)




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

batch_size = 100
num_epochs = 100
# lr0 = 0.1

h0 = 28*28    #784
# h1 = b1 = 500
# h2 = b2 = 300
h3 = b3 = 10
cuda = True

hp = [[5*787, 4*437, 77, 0.088607443091241619, 'glorot'],]

n = len(hp)

loss_train = np.empty([n,num_epochs]) 
loss_valid = np.empty([n,num_epochs]) 

acc_train = np.empty([n,num_epochs]) 
acc_valid = np.empty([n,num_epochs]) 
lr = np.empty([n,num_epochs]) 

t0 = time.time()        
for i, (h1s, h2s, bs, lrs, method_) in enumerate(hp):
    
    train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=bs, shuffle=True)
    valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=bs, shuffle=False)
    test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=bs, shuffle=False)

    model = MLPLinear([h0,h1s,h2s,h3],cuda) 
    model.initialization(method_)
    optimizer = optim.SGD(model.parameters(), lr=lrs)
    
    print('     ')
    print('     ')    
    print('Initialization method : '+method_)
    print('________________________________')    
    for j in range(num_epochs):
        model_loss = 0
        for batch_idx, (x,y) in enumerate(train_batch):
            xt,yt = model.input(x,y)
            pred_batch = model.forward(xt)
            
            optimizer.zero_grad()
            loss_batch = model.criterion(pred_batch, yt)
            loss_batch.backward()
            optimizer.step()
        
        xt_,yt_ = model.input(train_data,train_labels)
        xv_,yv_ = model.input(valid_data,valid_labels)
        pred_train_all = model.forward(xt_)
        pred_valid_all = model.forward(xv_)  
        
        loss_train[i,j] = model.criterion(pred_train_all,yt_).data[0]
        loss_valid[i,j] = model.criterion(pred_valid_all,yv_).data[0]

        acc_train[i,j]  = prediction(train_batch,model)
        acc_valid[i,j]  = prediction(valid_batch,model)
        
        if j%10 ==0: print('Epoch #'+str(j)+', Train loss = '+str(loss_train[i,j])+', Valid loss = '+str(loss_valid[i,j]))
        lr[i,j] = adjust_lr(optimizer,lrs, j+1, num_epochs)
    print('Train accuracy = '+str(prediction(train_batch,model))+'%')
    ts = time.time()        
    print('Time :  %.1f sec'%(ts-t0))
print('done!')


filename = 'SaveData/Q1/LearningCurve_best_HP_VeryBig'
with open(filename, 'wb') as f: 
    pickle.dump([loss_train, loss_valid, acc_train, acc_valid,hp], f)



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################



# #### GAP
# cuda  = True
# #hp = [[2*787, 2437, 77, 0.088607443091241619, 'glorot'],]
# h0 = 28*28
# h1 = 787
# h2 = 437
# h3 = 10
# lr0 = 0.0886
# num_epochs = 100
# batch_size = 77
# a = [0.01, 0.02, 0.05, 0.1, 1.0]
# T = 5

# initialization_method = ['glorot']

# loss_valid_all=np.empty([T,len(a),num_epochs])
# loss_train_all=np.empty([T,len(a),num_epochs])
# loss_test_all=np.empty([T,len(a),num_epochs])
# acc_valid_all=np.empty([T,len(a),num_epochs])
# acc_train_all=np.empty([T,len(a),num_epochs])
# acc_test_all=np.empty([T,len(a),num_epochs])


# #         ind = torch.randperm(int(a_*50000))

# for i in range(T):
#     print(str(i+1)+'/5')


#     for j,a_ in enumerate(a):


#         ind = torch.randperm(int(a_*50000))
#         train_data_subset = train_data[ind]
#         train_labels_subset = torch.FloatTensor(train_y[ind]).long()

#         train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data_subset,train_labels_subset), batch_size=batch_size, shuffle=True)
#         valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
#         test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)
        
#         model = MLPLinear([h0,h1,h2,h3],cuda) 
#         model.initialization('glorot')
#         optimizer = optim.SGD(model.parameters(), lr=lr0)
            
#         loss_train = np.array([])
#         loss_valid = np.array([])
#         loss_test = np.array([])

#         acc_train = np.array([])
#         acc_valid = np.array([])
#         acc_test = np.array([])
#         lr = np.array([])
        
#         for k in range(num_epochs):
#             for batch_idx, (x,y) in enumerate(train_batch):
#                 xt,yt = model.input(x,y)
#                 pred_batch = model.forward(xt)
            
#                 optimizer.zero_grad()
#                 loss_batch = model.criterion(pred_batch, yt)
#                 loss_batch.backward()
#                 optimizer.step()    
            
#             xt_,yt_ = model.input(train_data,train_labels)
#             xv_,yv_ = model.input(valid_data,valid_labels)
#             xte_,yte_ = model.input(test_data,test_labels)
#             pred_train_all = model.forward(xt_)
#             pred_valid_all = model.forward(xv_)  
#             pred_test_all = model.forward(xte_)  
        
            
#             loss_valid = np.append(loss_valid,model.criterion(pred_valid_all,yv_).data[0])
#             loss_train = np.append(loss_train,model.criterion(pred_train_all,yt_).data[0])
#             loss_test = np.append(loss_test,model.criterion(pred_test_all,yte_).data[0])
#             acc_valid  = np.append(acc_valid,prediction(valid_batch,model))
#             acc_train  = np.append(acc_train,prediction(train_batch,model))
#             acc_test  = np.append(acc_test,prediction(test_batch,model))
#             lr_ = adjust_lr(optimizer,lr0, j+1, num_epochs)
#             lr = np.append(lr,lr_)
              
#             if k%20 == 0:
#                 print('Epoch #'+str(k)+', Train loss = '+str(loss_train[k])+', Valid loss = '+str(loss_valid[k]))
        
#         print('Train accuracy = '+str(prediction(train_batch,model))+'%'+',  Valid accuracy = '+str(prediction(valid_batch,model))+'%')        
    
#         loss_train_all[i,j,:] = loss_train
#         loss_valid_all[i,j:] = loss_valid
#         loss_test_all[i,j:] = loss_test
#         acc_train_all[i,j:] = acc_train
#         acc_valid_all[i,j:] = acc_valid
#         acc_test_all[i,j:] = acc_test

# print('Done!')



# filename = 'Generalize_Gap_v4'
# with open(filename, 'wb') as f: 
#     pickle.dump([loss_train_all, loss_valid_all,loss_test_all, acc_train_all, acc_valid_all,acc_test_all,h1,h2,batch_size,lr0], f)


