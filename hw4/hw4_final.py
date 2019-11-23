#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("hw04_data_set.csv")


# In[4]:


y = df.iloc[:,1]
X = df.iloc[:,0]
X_train = df.iloc[:150,0].reset_index(drop = True)
X_test = df.iloc[150:,0].reset_index(drop = True)
y_train = df.iloc[:150,1].reset_index(drop = True)
y_test = df.iloc[150:,1].reset_index(drop = True)


# In[5]:


minX = min(X_train)
maxX = max(X_train)
miny = min(y)
maxy = max(y)


# In[6]:


point_colors = ['red','blue']
origin = 1.5
min_val = minX
max_val = maxX
N = len(y_train)
X_train = X_train.values
y_train = y_train.values
y_test = y_test.values
X_test = X_test.values


# In[7]:


bin_width = 0.37
left_borders = np.arange(origin, max_val,step = bin_width)
right_borders = np.arange(origin + bin_width,max_val+bin_width,step = bin_width)
data_interval = np.arange(origin,max_val,step = 0.001)


# In[8]:


def regressogram(b):
    sums = 0
    count = 0
    for i in range(0,len(X_train)):
        if(X_train[i] <= right_borders.T[b]):
            if(left_borders.T[b] < X_train[i]):
                sums +=  y_train[i]
                count += 1
    val = sums / (count)
    return val


# In[9]:


def running_mean_smoother(b):
    sums = 0
    count = 0
    bin_width = 0.37
    for i in range(0,len(X_train)):
        if(np.abs((data_interval[b]-X_train[i])/bin_width) < 1/2):
            sums +=  y_train[i]
            count += 1
    val = sums / count
    return val


# In[10]:


import math
def kernel_smoother(b):
    sums = 0
    counts = 0
    bin_width = 0.37
    
    for i in range(0,len(X_train)):
        u = (data_interval[b] - X_train[i]) / bin_width
        Gaussian_Kernel = 1/np.sqrt(2* math.pi) * np.exp(-1* u ** 2 / 2)
        counts += Gaussian_Kernel
        sums += (Gaussian_Kernel * y_train[i])
        
    val = sums / counts
    return val


# In[11]:


def RMSE(y_test,X_test,p_head_type,name):
    rmse = 0
    for i in range(0,len(X_test)):
            if name == "Regressogram":
                for b in range(0,len(left_borders)):
                    if(left_borders[b] < X_test[i] and X_test[i] <= right_borders[b]):
                        a = (y_test[i] - p_head_type[int((X_test[i]-origin)/bin_width)])**2
                        rmse += a
                result = np.sqrt(rmse / len(X_test))
            elif name == "Kernel Smoother":
                b = (y_test[i] - p_head_type[int((X_test[i]-origin)/0.001)])**2
                rmse += b
                result = np.sqrt(rmse / len(X_test))
            elif name == "Running Mean Smoother":
                c = (y_test[i] - p_head_type[int((X_test[i]-origin)/0.001)])**2
                rmse += c
                result = np.sqrt(rmse / len(X_test))
    print("{name} => RMSE is {rmse} when h is {h}".format(name = name,rmse = np.round(result,4), h = bin_width))


# In[12]:


p_head = []
for i in range(0,len(left_borders)):
    p_head.append(regressogram(i))  
p_head = np.asarray(p_head)


# In[13]:


dot1 = plt.scatter(X_train,y_train, c=point_colors[1], s=20, alpha=0.8)
dot2 = plt.scatter(X_test,y_test, c=point_colors[0], s=20, alpha=0.8)
plt.legend(['training','test'])

for b in range(1,len(left_borders)+1):
    #if b == 10:
     #   break
    plt.plot((left_borders[b-1],right_borders[b-1]),(p_head[b-1],p_head[b-1]),c = 'black')
    if b < len(left_borders):
        plt.plot((right_borders[b-1],right_borders[b-1]),(p_head[b-1],p_head[b]),c = 'black')


plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.title('{name} , h = {h}'.format(name = "Regressogram",h = bin_width))
plt.ylim = ((miny, maxy))
plt.xlim = ((minX  + bin_width, maxX + bin_width ))
plt.show()

# In[14]:


RMSE(y_test,X_test,p_head_type = p_head,name = "Regressogram")


# In[15]:


p_head_rm = []
for i in range(0,len(data_interval)):
    p_head_rm.append(running_mean_smoother(i))  
p_head_rm = np.asarray(p_head_rm)


# In[24]:


plt.scatter(X_train,y_train, c=point_colors[1], s=20, alpha=0.8)
plt.scatter(X_test,y_test, c=point_colors[0], s=20, alpha=0.8)
plt.legend(['training','test'])

plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.title('{name} , h = {h}'.format(name = "Running Mean Smoother",h = bin_width))
plt.ylim = ((miny, maxy))
plt.xlim = ((minX, maxX))

"""
for b in range(1,len(data_interval)):
    plt.plot((data_interval[b-1],data_interval[b]),(p_head_rm[b-1],p_head_rm[b-1]),c = 'black')
    if b < len(left_borders):
        plt.plot((data_interval[b-1],data_interval[b]),(p_head_rm[b-1],p_head_rm[b]),c = 'black')
"""
plt.plot(data_interval,p_head_rm, c = 'black')
plt.show()

# In[17]:


RMSE(y_test,X_test,p_head_type = p_head_rm,name = "Running Mean Smoother")


# In[18]:


p_head_ks = []
for i in range(0,len(data_interval)):
    p_head_ks.append(kernel_smoother(i))  
p_head_ks = np.asarray(p_head_ks)


# In[19]:


dot1 = plt.scatter(X_train,y_train, c=point_colors[1], s=20, alpha=0.8)
dot2 = plt.scatter(X_test,y_test, c=point_colors[0], s=20, alpha=0.8)
plt.legend(['training','test'])

plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.title('{name} , h = {h}'.format(name = "Kernel Smoother",h = bin_width))
plt.ylim = ((miny, maxy))
plt.xlim = ((minX  - bin_width, maxX + bin_width ))
plt.plot(data_interval,p_head_ks,color = 'black')
plt.show()

# In[20]:


RMSE(y_test,X_test,p_head_type = p_head_ks,name = "Kernel Smoother")


# In[ ]:




