#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 16 22:03:26 2021

@author: amir
"""
from numba import jit
from numba import cuda, float32
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mes
from sklearn.metrics import accuracy_score as acc

import time
import matplotlib.pyplot as plt
import math


sin_wave = np.array([math.sin(x) for x in np.arange(200)])
#plt.plot(sin_wave[:50])




X = []
Y = []

seq_len = 50
num_records = len(sin_wave) - seq_len

for i in range(num_records - 50):
    X.append(sin_wave[i:i+seq_len])
    Y.append(sin_wave[i+seq_len])
    
X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)



X_val = []
Y_val = []

for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i:i+seq_len])
    Y_val.append(sin_wave[i+seq_len])
    
X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)


learning_rate = 0.0001    
nepoch = 25               
T = 50                   # length of sequence
hidden_dim = 100         # تعداد نرون در لایه مخفی
output_dim = 1          # خروجی
bias = 0.1              # بایاس
bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

# وزن ها
U = np.random.uniform(0, 1, (hidden_dim, T)) # وزن های تصادفی بین ورودی و لایه مخفی 
V = np.random.uniform(0, 1, (hidden_dim, hidden_dim)) # وزن های تصادفی بین لایه مخفی و لایه خروجی
W = np.random.uniform(0, 1, (output_dim, hidden_dim)) # وزن های تصادفی مشترک 


#تابع سیکموئید
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanganthyperbolic(x):
    return math.tanh(x);



for epoch in range(nepoch):
    # check loss on train
    loss = 0.0
    
    # do a forward pass to get prediction
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]                    # get input, output values of each record
        prev_s = np.zeros((hidden_dim, 1))   # here, prev-s is the value of the previous activation of hidden layer; which is initialized as all zeroes
        for t in range(T):
            new_input = np.zeros(x.shape)    # we then do a forward pass for every timestep in the sequence
            new_input[t] = x[t]              # for this, we define a single input for that timestep
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s

    # calculate error 
        loss_per_record = (y - mulv)**2 / 2
        loss += loss_per_record
    loss = loss / float(y.shape[0])


    # check loss on val
    val_loss = 0.0
    for i in range(Y_val.shape[0]):
        x, y = X_val[i], Y_val[i]
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s

        loss_per_record = (y - mulv)**2 / 2
        val_loss += loss_per_record
    val_loss = val_loss / float(y.shape[0])

    print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)

    # train model
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
    
        layers = []
        prev_s = np.zeros((hidden_dim, 1))
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        dW = np.zeros(W.shape)
        
        dU_t = np.zeros(U.shape)
        dV_t = np.zeros(V.shape)
        dW_t = np.zeros(W.shape)
        
        dU_i = np.zeros(U.shape)
        dW_i = np.zeros(W.shape)
        
        # forward pass
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            layers.append({'s':s, 'prev_s':prev_s})
            prev_s = s


        # derivative of pred
        dmulv = (mulv - y)
        
        # backward pass
        for t in range(T):
            dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
            dsv = np.dot(np.transpose(V), dmulv)
            
            ds = dsv
            dadd = add * (1 - add) * ds
            
            dmulw = dadd * np.ones_like(mulw)

            dprev_s = np.dot(np.transpose(W), dmulw)


            for i in range(t-1, max(-1, t-bptt_truncate-1), -1):
                ds = dsv + dprev_s
                dadd = add * (1 - add) * ds

                dmulw = dadd * np.ones_like(mulw)
                dmulu = dadd * np.ones_like(mulu)

                dW_i = np.dot(W, layers[t]['prev_s'])
                dprev_s = np.dot(np.transpose(W), dmulw)

                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                dU_i = np.dot(U, new_input)
                dx = np.dot(np.transpose(U), dmulu)

                dU_t += dU_i
                dW_t += dW_i
                
            dV += dV_t
            dU += dU_t
            dW += dW_t

            if dU.max() > max_clip_value:
                dU[dU > max_clip_value] = max_clip_value
            if dV.max() > max_clip_value:
                dV[dV > max_clip_value] = max_clip_value
            if dW.max() > max_clip_value:
                dW[dW > max_clip_value] = max_clip_value
                
            
            if dU.min() < min_clip_value:
                dU[dU < min_clip_value] = min_clip_value
            if dV.min() < min_clip_value:
                dV[dV < min_clip_value] = min_clip_value
            if dW.min() < min_clip_value:
                dW[dW < min_clip_value] = min_clip_value
        
        # update
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW



preds = []
for i in range(Y_val.shape[0]):
    x, y = X_val[i], Y_val[i]
    prev_s = np.zeros((hidden_dim, 1))
    # For each time step...
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s

    preds.append(mulv)
    
preds = np.array(preds)

plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y_val[:, 0], 'r')
plt.show()




math.sqrt(mes(Y_val[:, 0] * max_val, preds[:, 0, 0] * max_val))


























column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
file = pd.read_csv('WISDM_at_v2.0_raw.txt', header=None, names=column_names, comment=';')
file = file.dropna(axis=0, how='any')
file = file[1:(1 * file.shape[0]) // 100]

file = file.drop(axis=1,columns=['user-id','timestamp'])


# اماده سازی دیتا

for x in range(1,(len(file)+1)):
    if(file.loc[x]['activity'] == 'Walking') :
        file.at[x,'activity'] = 1
    elif (file.loc[x]['activity'] == 'Jogging') :
        file.at[x,'activity'] = 2
    elif (file.loc[x]['activity'] == 'LyingDown') :
        file.at[x,'activity'] = 3
    elif (file.loc[x]['activity'] == 'Standing') :
        file.at[x,'activity'] = 4
    elif (file.loc[x]['activity'] == 'Sitting') :
        file.at[x,'activity'] = 5





data = file.to_numpy()








inputs = data[:,1:] # ورودی ها

outputs = data[:,0] # خروجی

epochs = 500 #  تعداد ایپک  
learningrate = 0.1 # نرخ اموزش


#### شافل کردن 
per_list =  np.random.permutation(len(data))

inputs_sh = []
outputs_sh = []

for i in range (len(data)):
    indexi = per_list[i]
    temp_input = inputs[i]
    temp_output = outputs[i]
    inputs_sh.append(temp_input)
    outputs_sh.append(temp_output)
    
inputs_sh = np.array(inputs_sh)
outputs_sh = np.array(outputs_sh) 


#minvec = inputs_sh.min(axis=0)
#maxvec = inputs_sh.max(axis=0)
#inputs_sh2 = (inputs_sh - minvec ) / ( maxvec - minvec)



## Scaling the features
scaler = MinMaxScaler(feature_range=(0,1))
## Transform the data into
inputs_sh = scaler.fit_transform(inputs_sh)


trn_test_split = int (0.75 * len(inputs_sh))

X_train = inputs_sh[0:trn_test_split, :]
Y_train = outputs_sh[0 :trn_test_split]

X_val = inputs_sh[trn_test_split : , :]
Y_val = outputs_sh[trn_test_split :, ]



n0 = 3 # تعداد نرورن در لایه ورودی

n1 = 3 #تعداد نرود در لایه دوم 

n2 = 3 # تعداد نرون در لایه سوم

n3 = 1 # تعداد نرون در لایه آخر




w1 = np.random.uniform(low=-10,high=+10,size=(n1,n0)) # وزن های تصادفی در لایه ورودی
w2 = np.random.uniform(low=-10,high=+10,size=(n2,n1)) # وزن های تصادفی در لایه دوم
w3 = np.random.uniform(low=-10,high=+10,size=(n3,n2)) # وزن های تصادفی در لایه سوم


@jit(nopython=True, fastmath=True) # Set "nopython" mode for best performance, equivalent to @njit
# تابع سیکموید
def activation(x):
    y = 1/(1 + np.exp(-1*x))
    return y


@jit(nopython=True, fastmath=True) # Set "nopython" mode for best performance, equivalent to @njit
# الگوریتم پیشرو
def feedforwardOneLayer(input_x , w):
    x = np.dot(input_x , w.T)
    y = activation(x)
    return y

@jit(nopython=True, fastmath=True)  # Set "nopython" mode for best performance, equivalent to @njit
# مشتق تابع سیکموید
def d_activation(y):
    d_y = y * ( 1 - y )    
    return d_y


#@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit(nopython=True, parallel=True)
def diagnal(y):
    d = d_activation(y)
    return np.diagflat(d)



list_accTrain = [] 
list_mesTrain = [] 
list_accValid = [] 
list_mesValid = []



start = time.time()
for i in range(epochs):
    for j in range(len(X_train)):
        inputNet = X_train[j]
        inputNet = np.reshape(inputNet,newshape=(1,n0))
        targetNet = Y_train[j]        
        y1 = feedforwardOneLayer(inputNet , w1)
        y2 = feedforwardOneLayer(y1 , w2)
        y3 = feedforwardOneLayer(y2 , w3)
        
        error = targetNet - y3
        
        diagf1 = diagnal(y1)
        diagf2 = diagnal(y2)
        diagf3 = d_activation(y3)
        
        # w1 = w1 - lr * (-2/N)*(error) * d_f3 * w3 * d_f2 * w2 * d_f1 * ...
        # ... * input
        
        temp2 = -2 * error * diagf3
        temp1 = temp2 * w3
        temp1 = np.dot( temp1 , diagf2 )
        temp = np.dot( temp1 , w2 ) 
        temp = np.dot( temp , diagf1)
        temp = temp.T
        temp = np.dot( temp , inputNet)
        
        w1 = w1 - learningrate * temp
        
        w2 = w2 - learningrate * np.dot( temp1.T , y1 )
        
        w3 = w3 - learningrate * np.dot( temp2.T , y2)
        
    NetOut_train =[]
    Target_train = []
    Rnd_Netoutput_Train = []
    for x in range(len(X_train)):
        inputNet = X_train[x]
        target = Y_train[x]
        
        Target_train.append(target)
        y1 = feedforwardOneLayer(inputNet , w1)
        y2 = feedforwardOneLayer(y1 , w2)
        y3 = feedforwardOneLayer(y2 , w3)
        
        NetOut_train.append(y3)
        Rnd_Netoutput_Train.append(np.round(y3))
    
    mes_train = mes(Target_train , NetOut_train)
    list_mesTrain.append(mes_train)
    acc_train = acc(Target_train ,  Rnd_Netoutput_Train)
    list_accTrain.append(acc_train)
    
    print('epoch' , i , ' MSE_Train = ' , mes_train , '\tAcc_train = ', acc_train)
    
end = time.time()    

print("Elapsed (after compilation) = %s" % (end - start))




plt.figure(figsize=(20,10))
plt.plot(list_accTrain, color = 'green', label = 'accuracy Train')
plt.plot(list_mesTrain, color = 'red', label = '')
plt.title('Prediction')
plt.xlabel('Epoch')
#plt.ylabel(' ')
plt.legend()
plt.show()