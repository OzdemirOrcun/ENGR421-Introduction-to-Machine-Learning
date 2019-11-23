#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


labels_df = pd.read_csv("hw03_labels.csv",header = None)
images_df = pd.read_csv("hw03_images.csv",header = None)
initial_W_df = pd.read_csv("initial_W.csv",header = None)
initial_v_df = pd.read_csv("initial_v.csv",header = None)


# In[3]:


from sklearn.preprocessing import LabelEncoder
y_encoder = LabelEncoder()
labels_df = y_encoder.fit_transform(labels_df)
labels_df = pd.DataFrame(labels_df)


# In[4]:


combined_df = pd.concat([images_df, labels_df], axis=1)


# In[5]:


X = combined_df.iloc[:,:-1]
y = combined_df.iloc[:,-1]
X_train = X.iloc[:500,:]
X_test = X.iloc[500:,:]
y_train = y.iloc[:500]
y_test = y.iloc[500:]


# In[6]:


def sigmoid(X,W):
    sigmoid = (1/(1 + np.exp((-1)*(np.dot(X,W)))))
    return sigmoid


# In[7]:


def softmax(Z,v):
    a = (np.exp(np.dot(Z,v)))
    b = np.sum(np.exp(np.dot(Z,v)),axis = 1).reshape(len(a),1)

    return a/b


# In[8]:


def safelog(x):
    return np.log(x + 1e-100)


# In[9]:


def ones_column_binder(ones,data):
    data = pd.DataFrame(data)
    data = pd.concat([ones,data],axis = 1)
    data = np.asarray(data)
    return data


# In[10]:


import random
import math
H = 20
eta = 0.0005
epsilon = 1e-3
max_iteration = 500


# In[11]:


W = initial_W_df.values
v = initial_v_df.values


# In[12]:


Identity_Matrix = X_train.iloc[:,0]


# In[13]:


y_train_array = np.asarray(y_train)
y_train_array = y_train_array.reshape(500,1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories = 'auto')
encoded_y_train = onehotencoder.fit_transform(y_train_array).toarray()


# In[14]:


iteration = 1
objective_values = []
while 1:
    Z = sigmoid(ones_column_binder(Identity_Matrix,X_train),W)
    y_pred = softmax(ones_column_binder(Identity_Matrix,Z),v)

    objective_values.append(-np.sum(encoded_y_train * safelog(y_pred)))


    delta_v = np.dot( ones_column_binder(Identity_Matrix,Z).T,(encoded_y_train - y_pred))

    
    
    first_part = np.sum(np.dot((encoded_y_train - y_pred) , v.T),axis = 1).reshape((500,1))

   
    first_part = first_part * Z * (1 - Z)
    second_part = ones_column_binder(Identity_Matrix,X_train).T
    delta_W =  np.dot(second_part, first_part)

    v = v + delta_v * eta  

    W = W + eta * delta_W 



    print("Iteration #{n}".format(n = iteration))

    if(iteration >= max_iteration):
        break

        
    iteration += 1


# In[15]:


plt.xlabel("Iteration")
plt.ylabel("Error")
plt.plot( 
  objective_values,
  color = 'black'
)


# In[16]:


y_train_pred = []

tr_rows = len(X_train.index)

for i in range(0,tr_rows):
    y_train_pred.append(np.argmax(y_pred[i]))
y_train_pred = np.asarray(y_train_pred)


# In[17]:


y_test_array = np.asarray(y_test)
y_test_array = y_test_array.reshape(500,1)


# In[18]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder_ = OneHotEncoder(categories = 'auto')
encoded_y_test = onehotencoder_.fit_transform(y_test_array).toarray()


# In[19]:


Identity_Matrix_t = X_test.iloc[:,0]
X_u = ones_column_binder(Identity_Matrix_t,X_test)

Z1 = sigmoid(X_u,W)

y_pred_test = softmax(ones_column_binder(Identity_Matrix,Z1),v)

y_test_pred = []

te_rows = len(X_test.index) 

for i in range(0,te_rows):
    y_test_pred.append(np.argmax(y_pred_test[i]))
y_test_pred = np.asarray(y_test_pred)


# In[20]:


from sklearn.preprocessing import LabelEncoder
y_encoder = LabelEncoder()
categorized_train_y_for_cm = y_encoder.fit_transform(y_train)
categorized_test_y_for_cm = y_encoder.fit_transform(y_test)


# In[21]:


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(categorized_train_y_for_cm,y_train_pred)
cm_test = confusion_matrix(categorized_test_y_for_cm,y_test_pred)


# In[22]:


cm_train = cm_train.T
cm_test = cm_test.T


# In[23]:


print("Confusion Matrix of Train Data.\n")
print(cm_train)
print("\n")


# In[24]:


print("Confusion Matrix of Test Data.\n")
print(cm_test)
print("\n")


# In[25]:


#from sklearn.metrics import f1_score
#f1_score(categorized_test_y_for_cm,y_test_pred, average='macro')  

#0.9329753195502425


# In[26]:


#from sklearn.metrics import f1_score
#f1_score(categorized_train_y_for_cm,y_train_pred, average='macro')  

#0.9553509566328586


# In[ ]:




