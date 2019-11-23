#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


labels_df = pd.read_csv("hw02_labels.csv",header = None)
images_df = pd.read_csv("hw02_images.csv",header = None)
initial_W_df = pd.read_csv("initial_W.csv",header = None)
initial_w0_df = pd.read_csv("initial_w0.csv",header = None)


# In[3]:


#labels_df.head()


# In[4]:


#images_df.head()


# In[5]:


#initial_W_df.shape


# In[6]:


#initial_w0_df.shape


# In[7]:


combined_df = pd.concat([images_df, labels_df], axis=1)


# In[8]:


X = combined_df.iloc[:,:-1]


# In[9]:


#X.head()


# In[10]:


y = combined_df.iloc[:,-1]


# In[11]:


#combined_df.values


# In[12]:


X_train = X.iloc[:500,:]
X_test = X.iloc[500:,:]


# In[13]:


y_train = y.iloc[:500]
y_test = y.iloc[500:]


# In[14]:


def sigmoid(X,W,w0):
    sigmoid = (1/(1 + np.exp(-1*(np.dot(X,W)+w0))))
    return sigmoid


# In[15]:


def update_W(X,y_truth,y_pred):
    a = (y_truth - y_pred) * y_pred * (1 - y_pred)
    return (np.dot(-1*(np.transpose(X)),a))


# In[16]:


def update_w0(y_truth,y_pred):
    a =((y_truth - y_pred) * y_pred * (1- y_pred))
    col_sums = a.sum(axis = 0)
    return -1 * col_sums


# In[17]:


import math,random
epsilon = 1e-3
eta = 0.0001
random.seed(54020)


# In[18]:


#X_train.shape


# In[19]:


W = initial_W_df.values
w0 = initial_w0_df.values.reshape(5,)


# In[20]:


#w0.shape


# In[21]:


#W.shape


# In[22]:


y_train_array = np.asarray(y_train)
y_train_array = y_train_array.reshape(500,1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories = 'auto')
encoded_y_train = onehotencoder.fit_transform(y_train_array).toarray()


# In[23]:


#encoded_y_train


# In[24]:


objective_values = []
iteration = 1
while 1:
    training_Y = sigmoid(X_train,W,w0)
    objective_values.append(np.sum((encoded_y_train - training_Y)**2)*(0.5))
    #objective_values.append(-1 * np.sum(encoded_y_train * np.log(training_Y) + (1 - encoded_y_train)* (1 - np.log(training_Y) )))
    prew_W = W
    W = W - eta * update_W(X_train,encoded_y_train,training_Y)
    
    prew_wo = w0
    w0 = w0 - eta*update_w0(encoded_y_train,training_Y)
    
    q = np.sqrt(np.sum((w0 - prew_wo)**2) + np.sum((W - prew_W)**2))
    
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.plot( 
      objective_values,
      color = 'black'
    )
    
    if(iteration >= 500):
        break
    
    if(epsilon > q):
        break
    
    
    iteration += 1


# In[25]:


#training_Y


# In[26]:


y_train_pred = []

tr_rows = len(X_train.index)

for i in range(0,tr_rows):
    y_train_pred.append(np.argmax(training_Y[i]))
y_train_pred = np.asarray(y_train_pred)


# In[27]:


#y_test.shape


# In[28]:


y_test_array = np.asarray(y_test)
y_test_array = y_test_array.reshape(500,1)


# In[29]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder_ = OneHotEncoder(categories = 'auto')
encoded_y_test = onehotencoder_.fit_transform(y_test_array).toarray()


# In[30]:


y_pred_test = sigmoid(X_test,W,w0)
y_test_pred = []

te_rows = len(X_test.index) 

for i in range(0,te_rows):
    y_test_pred.append(np.argmax(y_pred_test[i]))
y_test_pred = np.asarray(y_test_pred)


# In[31]:


from sklearn.preprocessing import LabelEncoder
y_encoder = LabelEncoder()
categorized_train_y_for_cm = y_encoder.fit_transform(y_train)
categorized_test_y_for_cm = y_encoder.fit_transform(y_test)


# In[32]:


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(categorized_train_y_for_cm,y_train_pred)
cm_test = confusion_matrix(categorized_test_y_for_cm,y_test_pred)


# In[33]:


cm_train = cm_train.T


# In[34]:


cm_test = cm_test.T


# In[35]:


print(cm_train)


# In[36]:


print(cm_test)

