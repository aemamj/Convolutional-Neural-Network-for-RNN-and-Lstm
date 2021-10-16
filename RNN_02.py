#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 30 22:03:26 2021

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


## Scaling the features
scaler = MinMaxScaler(feature_range=(0,1))
## Transform the data into
inputs_sh = scaler.fit_transform(inputs)




lenght = 50 # مقدار اندازه برگشت به عقب
num_records = len(inputs) - lenght

X = inputs_sh.copy()
Y = outputs.copy()
'''
for i in range(num_records):
    X.append(inputs_sh[i:i+lenght])
    Y.append(outputs[i+lenght])
'''
X = np.array(X)

Y = np.array(Y)




epochs = 500 #  تعداد ایپک  
learningrate = 0.1 # نرخ اموزش

T = num_records                   # length of sequence
hidden_dim = 50         # تعداد نرون در لایه مخفی
output_dim = 1          # خروجی
bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10


# وزن ها

biasy = np.random.uniform(0, 1, )              # بایاس
biash = np.random.uniform(0, 1, )              # 

Wh = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی بین ورودی و لایه مخفی 
Wx = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی بین لایه مخفی و لایه خروجی
Wy = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی مشترک 


#تابع سیکموئید
def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))
# تابع تانژانت هیپربولیک
def tanganthyperbolic(x):
    return np.tanh(x);

# تابع نمایی نرمال
def softmax(x):
    return np.exp(x) / np.exp(x)

# اتابع پیشرو
def forward(wh,wx,wy,bh,by,h,x):
    inputh = wh*h
    inputx = wx*x
    s = inputx+inputh+bh
    outputh = tanganthyperbolic(s)
    yt = np.dot(outputh , wy ) + by
    #outputy = softmax(yt)  
    return yt , outputh


def backward(wh,wx,wy,bh,by,x,h,h_prev,dh):
    dtanh = (1-h**2) * dh
    
    dx = wx * dtanh
    dwx = dtanh * x    
    
    d_prev = wy * dtanh
    dwh = dtanh * h_prev 
    
    dby = dtanh + 1
    
    return dby , dwx , dwh 


list_accTrain = [] 
list_mesTrain = [] 
list_accValid = [] 
list_mesValid = []

start = time.time()
for epoch in range(epochs):
    # check loss on train
    loss = 0.0
    
    # do a forward pass to get prediction
    prev_h = np.zeros((len(X), len(X[0])))
    for t in range(len(X)):
        y , prev_h[t] = forward(Wh[t],Wx[t],Wy[t],biash,biasy,prev_h[t-1],X[t])


    Dh = np.random.uniform(0, 1, (len(X)+1, len(X[0]) )) # وزن های تصادفی مشترک 
    for t in reversed(range(len(X))):
         Wy[t] , Wx[t] ,  prev_h[t-1] = backward(Wh[t],Wx[t],Wy[t],biash,biasy,X[t],prev_h[t],prev_h[t-1],Dh[x])
     
         
    #print('Epoch: ', epoch )
    
    NetOut_train =[]
    Target_train = []
    Rnd_Netoutput_Train = []
    for x in range(len(X)):
        inputNet = X[x]
        target = Y[x]
        
        y , prev_h[x] = forward(Wh[x],Wx[x],Wy[x],biash,biasy,prev_h[x-1],X[x])
    
    
        Target_train.append(target)
        
        NetOut_train.append(y)
        Rnd_Netoutput_Train.append(np.round(y))
    
    mes_train = mes(Target_train , NetOut_train)
    list_mesTrain.append(mes_train)
    acc_train = acc(Target_train ,  Rnd_Netoutput_Train)
    list_accTrain.append(acc_train)
    
    print('epoch' , epoch , ' MSE_Train = ' , mes_train , '\tAcc_train = ', acc_train)
    
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
