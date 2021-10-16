#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 16 11:19:43 2021

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


@jit(nopython=True, fastmath=True) 
# تابع سیکموید
def activation(x):
    y = 1/(1 + np.exp(-1*x))
    return y


@jit(nopython=True, fastmath=True) 
# الگوریتم پیشرو
def feedforwardOneLayer(input_x , w):
    x = np.dot(input_x , w.T)
    y = activation(x)
    return y

@jit(nopython=True, fastmath=True)  
# مشتق تابع سیکموید
def d_activation(y):
    d_y = y * ( 1 - y )    
    return d_y


#@jit(nopython=True)
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