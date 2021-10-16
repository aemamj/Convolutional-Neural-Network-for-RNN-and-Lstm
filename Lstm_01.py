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

bias0 = np.random.uniform(0, 1, )              # بایاس
bias1 = np.random.uniform(0, 1, )              # 
bias2 = np.random.uniform(0, 1, )              # بایاس
bias3 = np.random.uniform(0, 1, )              # 
bias4 = np.random.uniform(0, 1, )              # بایاس
biasy = np.random.uniform(0, 1, )  


W0 = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی بین ورودی و لایه مخفی 
W1 = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی بین لایه مخفی و لایه خروجی
W2 = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی مشترک 
W3 = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی مشترک 
W4 = np.random.uniform(0, 1, (len(X), len(X[0]) )) # وزن های تصادفی مشترک 
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

# اتابع وردی فراموشی
def forgat_gate(wh,wx,b,h,x):
    inputh = wh*h
    inputx = wx*x
    y = inputx+inputh+b
    f = sigmoid(y)
    return f


# اتابع وردی فراموشی
def update_gate(wh,wx,b,h,x):
    inputh = wh*h
    inputx = wx*x
    y = inputx+inputh+b
    g1 = sigmoid(y)
    return g1



# اتابع وردی فراموشی
def tanh_gate(wh,wx,b,h,x):
    inputh = wh*h
    inputx = wx*x
    y = inputx+inputh+b
    g = tanganthyperbolic(y)
    return g


# اتابع وردی فراموشی
def G_Vector(update_gate , tanh_gate):
    return update_gate * tanh_gate

# اتابع وردی فراموشی
def Ct_Vector(c0,f):
    return c0*f


# اتابع وردی فراموشی
def C_output_Vector(Ct,G):
    return Ct+G


# اتابع وردی فراموشی
def Ch_Vector(Cout):
    return tanganthyperbolic(Cout)



# اتابع گیت خروجی
def O_Vector(wh,wx,b,h,x):
    inputh = wh*h
    inputx = wx*x
    y = inputx+inputh+b
    o = sigmoid(y)
    return o


#  مشتق تابع گیت خروجی
def d_O_Vector(wh,wx,b,h,x):
    inputh = wh*h
    inputx = wx*x
    y = inputx+inputh+b
    o = d_sigmoid(y)
    return o



# اتابع وردی فراموشی
def h_output_Vector(Ch,O):
    return O+Ch


###### backward ##########

@jit(nopython=True, fastmath=True)  
# مشتق تابع سیکموید
def d_sigmoid(y):
    d_y = y * ( 1 - y )    
    return d_y





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
    prev_c = np.zeros((len(X), len(X[0])))
    NetOut_train = []
    
    for t in range(len(X)):
        f  = forgat_gate(W0[t],W1[t],bias0,prev_h[t-1],X[t])
        g1 = update_gate(W2[t],W2[t],bias2,prev_h[t-1],X[t])
        g0 = forgat_gate(W3[t],W3[t],bias3,prev_h[t-1],X[t])
        g  = G_Vector(g1,g0)
        ct = Ct_Vector(prev_c[t-1],f)
        prev_c[t] = C_output_Vector(ct, g)
        ch = Ch_Vector(prev_c[t])
        o = O_Vector(W4[t], W4[t], bias4 , prev_h[t-1],X[t])
        prev_h[t] = h_output_Vector(ch , o)
        y = np.dot(prev_h[t] , Wy[t] ) + biasy
        NetOut_train.append(y)
        
    Dh = np.random.uniform(0, 1, (len(X)+1, len(X[0]) )) # وزن های تصادفی مشترک 
    for t in reversed(range(len(X))):
          error = Y[t] -  NetOut_train[t]
          d_o = error * tanganthyperbolic(prev_c[t])
          d_cs = error * d_o * ( 1 -( tanganthyperbolic(prev_c[t])**2 ))
          #d_y = error * d_o* ( 1 -( tanganthyperbolic(d_ch)**2 ))
          #d_g1 = error * d_o* ( 1 -( tanganthyperbolic(d_ch)**2 )) * g
          #d_g2 = error * d_o * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * i 
          d_f = error * d_o * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * prev_c[t-1] 
          d_ct = error * d_o * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_f

          
          dwxo = error * ( tanganthyperbolic(prev_c[t]) * d_O_Vector(W4[t], W4[t], bias4 , prev_h[t-1],X[t]) * X[t]
          dwho = error * ( tanganthyperbolic(prev_c[t]) * d_O_Vector(W4[t], W4[t], bias4 , prev_h[t-1],X[t]) * prev_h[t-1]
          dwbo = error * ( tanganthyperbolic(prev_c[t]) * d_O_Vector(W4[t], W4[t], bias4 , prev_h[t-1],X[t])

                   
          dwxf = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_sigmoid(zf) * xt
          dwhf = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_sigmoid(zf) * ht-1
          dwbo = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_sigmoid(zf) *
                          
                    
          dwxi = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_sigmoid(zi) * xt
          dwhi = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_sigmoid(zi) * ht-1
          dwbi = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * d_sigmoid(zi) *
                          
                    
          dwxg = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * sigmoid()
          dwhg = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * sigmoid()
          dwbg = error * ( 1 -( tanganthyperbolic(prev_c[t])**2 )) * sigmoid()     
                                         
                          
                          
                          
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
