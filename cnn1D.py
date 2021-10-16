#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:45:41 2021

@author: amir
"""

# Imports PIL module 
from PIL import Image
import enum
import numpy as np
import pandas as pd

from numba import jit
from numba import cuda, float32

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mes
from sklearn.metrics import accuracy_score as acc

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


class Pooling(enum.Enum):
   Max = 1
   Min = 2
   Avrage = 3
   Xmode = 4


class Filter(enum.Enum):
   Simpler_box_blur_1D = 1
   Gaussian_blur_1D = 2
   Line_detection_horizontal_1D  = 3
   Line_detection_vertical_1D = 4
   Line_detection_45D_1D = 5
   Line_detection_135D_1D = 6
   Line_detection_1D = 7
   Sobel_edge_horizontal_1D = 8
   Sobel_edge_vertical_1D = 9
   Laplacian_operator_1D = 10
   Laplacian_1D = 11
   Laplacian_gaussian_1D = 12

'''
xxx = X[:,0]


avarage = 0
for i in range(3):
    print(i)
    avarage += xxx[i]
avarage = avarage / 3
xxx = [avarage]+xxx
'''


def padding1D (data,pad) :
    print('count',len(data))
    print('pad',pad)
    avarage = 0
    for i in range(pad):
        print(i)
        avarage += data[i]
    avarage = avarage / pad
    for i in range(pad):
        data = [avarage]+data
    avarage = 0
    for i in range(pad):
        avarage += data[len(data)-i-1]
    avarage = avarage / pad
    for i in range(pad):
        np.append(data, avarage)
    print('count',len(data))
    return data



def Avragepooling1D(data,x,Pool):

    _data = []
            
    for i in reversed(range(1,Pool)):
        _data.append(data[x+i])
        _data.append(data[x-i])
    
    maxpool = (sum(_data)/len(_data))
    
    return maxpool




    
    
def conv1DPadding(data,pooling,Padding):
    print("Start" )
    if Padding > 0 :
        data = padding1D(data,Padding)
    
    X = len(data)
    ex = np.zeros(X)
        
    ListClum = []
    ListDeClum  = []


    for i in range(Padding,X,Padding):
        if  i > 0 and i < X-1   :
            #print(j)
            if i not in ListClum:
                ListClum.append(i)
            if pooling == Pooling.Avrage :
                ex[i] = Avragepooling1D(data,i,Padding)
                    
    for i in range(X):
        if  i < X-1   :
            if i not in ListClum:
                if i not in ListDeClum:
                    ListDeClum.append(i)
                    

    for i in range(0,1):
        if i not in ListClum:
            if i not in ListDeClum:
                ListDeClum.append(i)

                
    for i in range(X-1,X-Padding,-1):
        if i not in ListClum:
            if i not in ListDeClum:
                ListDeClum.append(i)
                

                
    
    a_del = np.delete(ex,np.array(ListDeClum), 1)
    

    return a_del



#out = Image.new(b.mode ,image.size)



#l1 = conv2d(image,Pooling.Xmode,1)
#l2 = conv2d(l1,Pooling.Xmode,1)
#l3 = conv2d(l2,Pooling.Avrage,5)


#l3.show()


#Simple box blur

    
def conv1DFiltter(data,filtring,Padding):
    print("StartFilter")

    X = len(data)
    
    print(X)
    
    if Padding > 0 :
        data = padding1D(data,Padding)


    
    
    ex = np.zeros(X)
        

    ListDeClum  = []
    ListClum = []
    

    
    for i in range(Padding,X):
    
        if i > Padding  and i < X-Padding  :            
            if i not in ListClum:
                ListClum.append(i)
                
            if filtring == Filter.Simpler_box_blur_1D :
                ex[i] = Simple_box_blur_1D(data,i)                
            if filtring == Filter.Line_detection_horizontal_1D :
                ex[i] = Line_detection_horizontal_1D(data,i)
            if filtring == Filter.Line_detection_vertical_1D :
                ex[i] = Line_detection_vertical_1D(data,i)
            if filtring == Filter.Line_detection_45D_1D :
                ex[i] = Line_detection_45D_1D(data,i)
            if filtring == Filter.Line_detection_135D_1D :
                ex[i] = Line_detection_135D_1D(data,i)                    
            if filtring == Filter.Line_detection_1D :
                ex[i] = Line_detection_1D(data,i)
                
            if filtring == Filter.Sobel_edge_horizontal_1D :
                ex[i] = Sobel_Edge_horizontal_1D(data,i)
            if filtring == Filter.Sobel_edge_vertical_1D :
                ex[i] = Sobel_Edge_vertical_1D(data,i)
                
            if filtring == Filter.Laplacian_1D :
                ex[i] = laplacian_1D(data,i)
                
    for i in range(X):
        if i not in ListClum:
            if i not in ListDeClum:
                ListDeClum.append(i)
                
    if 0 not in ListClum:        
        if 0 not in ListDeClum:
            ListDeClum.append(0)
        
    if X-1 not in ListClum:        
        if X-1 not in ListDeClum:            
            ListDeClum.append(X-1)
    
    a_del = np.delete(ex,np.array(ListDeClum))

    print("StartFilter")

    print(X)
    
    return a_del



"""
a =np.array([0.2111111111111111,0.2111111111111111,0.1121111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111])

#COL  =[[c[1][1],c[1][2],c[1][3]],[c[2][1],c[2][2],c[2][3]],[c[3][1],c[3][2],c[3][3]]]
b  =np.array([1,1,1,1,1,1,1,1,1])


A =  np.sum(a * b) 
#A = SBF[0][0] +SBF[0][1] +SBF[0][2] +SBF[1][0] +SBF[1][1] +SBF[1][2] +SBF[2][0] +SBF[2][1] +SBF[2][2] 

filtermirorr = np.flip(a)


"""


def Line_detection_135D_1D(data,x):
    
    FILTER =np.array([2,-1,-1,-1,2,-1,-1,-1,2])
    
    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
    
    return SubColor

def Line_detection_45D_1D(data,x):
    
    FILTER =np.array([-1,-1,2,-1,2,-1,2,-1,-1])
    
    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
    
    return SubColor

def Line_detection_vertical_1D(data,x):
    
    FILTER =np.array([-1,2,-1,-1,2,-1,-1,2,-1])
    
    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
    
    return SubColor

def Line_detection_horizontal_1D(data,x):
    
    FILTER =np.array([-1,-1,-1,2,2,2,-1,-1,-1])
    
    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
    
    return SubColor

def Line_detection_1D(data,x):
    
    FILTER = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1])

    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
        
    return SubColor

def Sobel_Edge_horizontal_1D(data,x):
    
    FILTER = np.array([-1,-2,-1,0,0,0,1,2,1])

    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
        
    return SubColor

def Sobel_Edge_vertical_1D(data,x):
    
    FILTER = np.array([-1,0,1,-2,0,2,-1,0,1])

    DATA  = np.array([data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
        
    return SubColor

def laplacian_1D(data,x):
    
    FILTER = np.array([-1,-1,1,-1,8,-1,-1,-1,1])

    DATA  = np.array([data[x-4],data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3],data[x+4]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
        
    return SubColor

def Simple_box_blur_1D(data,x):
    
    FILTER = np.array([0.1111111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111,0.1111111111111111, 0.1111111111111111,0.1111111111111111,0.1111111111111111])
    
    #print(data[x-4],data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3],data[x+4])
    DATA  = np.array([data[x-4],data[x-3],data[x-2],data[x-1],data[x],data[x+1],data[x+2],data[x+3],data[x+4]]  )  
    
    filtermirorr = np.flip(FILTER )
    SubColor = np.sum(FILTER  * DATA ) 
    
    return SubColor

#l1 = conv2dPadding(image,Pooling.Avrage,5)
l1 = conv1DFiltter(X[:,0],Filter.Laplacian_1D,4)
#l2 = conv1DFiltter(l1,Filter.Simpler_box_blur,1)
#l3 = conv1DFiltter(l2,Filter.Simpler_box_blur,1)
#l4 = conv1DFiltter(l3,Filter.Simpler_box_blur,1)
#l5 = conv1DFiltter(l4,Filter.Simpler_box_blur,1)
#l6 = conv1DFiltter(l5,Filter.Simpler_box_blur,1)
#l7 = conv1DFiltter(l6,Filter.Simpler_box_blur,1)
#l8 = conv1DFiltter(l7,Filter.Simpler_box_blur,1)
#l9 = conv1DFiltter(l8 ,Filter.Laplacian,1)
#l2.show()
#l9.show()
#image.show()





plt.figure(figsize=(20,10))
plt.plot(X, color = 'green', label = 'SBI Stock Price')
plt.plot(l1, color = 'red', label = 'Predicted SBI Stock Price')
plt.title('SBI Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Price')
plt.legend()
plt.show()




"""

import numpy as np 
a = np.array([[1,2,3],[0,2,5],[2,5,-1]]) 

print ("Array a:")
print (a) 


print ('Inverse of a:' )
print (np.flip(a, (0,1)))

a = np.array([[1,1, 1],
              [1, 1, 1],
              [1, 1, 1]])
b = np.array([[2, 1, 2],
              [4, 1, 1],
              [2, 1, 2]])

np.matmul(a, b)

np.sum(a)
np.sum(a[:, :] * b[: , :])
"""
