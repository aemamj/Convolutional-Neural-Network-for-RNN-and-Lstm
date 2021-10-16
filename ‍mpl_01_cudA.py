#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 22:03:26 2021

@author: amir
"""
from numba import jit
from numba import cuda
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mes
from sklearn.metrics import accuracy_score as acc
import time

import matplotlib.pyplot as plt



print(cuda.gpus)
print(cuda.cudadrv.devices.gpus)


file = pd.read_excel('cancer_.xlsx')

data = file.to_numpy()


inputs = data[:,:9]

outputs = data[:,9]

epochs = 350
learningrate = 0.001 

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



n0 = 9

n1 = 8 

n2 = 4

n3 = 1




w1 = np.random.uniform(low=-10,high=+10,size=(n1,n0))
w2 = np.random.uniform(low=-10,high=+10,size=(n2,n1))
w3 = np.random.uniform(low=-10,high=+10,size=(n3,n2))


@jit(nopython=True, fastmath=True) # Set "nopython" mode for best performance, equivalent to @njit
def activation(x):
    y = 1/(1 + np.exp(-1*x))
    return y


@jit(nopython=True, fastmath=True) # Set "nopython" mode for best performance, equivalent to @njit
def feedforwardOneLayer(input_x , w):
    x = np.dot(input_x , w.T)
    y = activation(x)
    return y

@jit(nopython=True, fastmath=True)  # Set "nopython" mode for best performance, equivalent to @njit
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


@jit(nopython=True, fastmath=True) 
def tm(error,diagf3,diagf2,diagf1):
    temp2 = -2 * error * diagf3
    temp1 = temp2 * w3
    temp1 = np.dot( temp1 , diagf2 )
    temp = np.dot( temp1 , w2 ) 
    temp = np.dot( temp , diagf1)
    temp = temp.T
    temp = np.dot( temp , inputNet)
    return temp , temp2 , temp1
    

    
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
        

        temp , temp2 , temp1 = tm(error,diagf3,diagf2,diagf1)

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
    

    NetOut_val =[]
    Target_val = []
    Rnd_Netoutput_val = []
    for x in range(len(X_val)):
        inputNet = X_val[x]
        target = Y_val[x]
        
        Target_val.append(target)
        y1 = feedforwardOneLayer(inputNet , w1)
        y2 = feedforwardOneLayer(y1 , w2)
        y3 = feedforwardOneLayer(y2 , w3)
        
        NetOut_val.append(y3)
        Rnd_Netoutput_val.append(np.round(y3))
    
    mes_val = mes(Target_val , NetOut_val)
    list_mesValid.append(mes_val)
    acc_val = acc(Target_val ,  Rnd_Netoutput_val)
    list_accValid.append(acc_val)
    
    print('epoch' , i , ' MSE_Train = ' , mes_train , '\tAcc_train = ', acc_train , ' MSE_Val = ' , mes_val , '\tAcc_Val = ', acc_val)
    
    
end = time.time()    

print("Elapsed (after compilation) = %s" % (end - start))   
    
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.title('MES')
plt.plot(list_mesTrain , label='mis_Train')
plt.plot(list_mesValid , label='mis_val')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('mes')


plt.subplot(2,1,2)
plt.title('Acc')
plt.plot(list_accTrain , label='acc_Train')
plt.plot(list_accValid , label='acc_val')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('acc')

plt.show()


