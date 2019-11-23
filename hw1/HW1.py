#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[52]:


images_df = pd.read_csv("./hw01_images.csv",header=None)
labels_df = pd.read_csv("./hw01_labels.csv",header=None,names = ['y'])


# In[53]:


combined_df = pd.concat([images_df, labels_df], axis=1)


# In[54]:


combined_df.shape


# In[55]:


X = combined_df.iloc[:,:-1]


# In[56]:


X.head()


# In[57]:


y = combined_df.iloc[:,-1]


# In[58]:


y.size


# In[59]:


y.dtype


# In[60]:


combined_df.values[200][4096]


# In[61]:


combined_df.head()


# In[62]:


x_train = X.iloc[0:200]
y_train = y.iloc[0:200]


# In[63]:


x_test = X.iloc[200:400]
y_test = y.iloc[200:400]


# In[64]:


"""y_train_a.head()"""
y_train


# In[65]:


x_train.shape


# In[66]:


#combined_df


# In[67]:


combined_df['y'].shape


# In[68]:


height = 400
width = 1

y_male= pd.DataFrame(0, index=range(height), columns=range(width))
y_female= pd.DataFrame(0, index=range(height), columns=range(width))

for i in range(0,len(combined_df)):
    if(combined_df['y'].values[i] == 1):
        y_female[i] = combined_df['y'].values[i]
    elif(combined_df['y'].values[i] == 2):
        y_male[i] = combined_df['y'].values[i]
male_list = []
female_list = []
for col in y_female.columns: 
    female_list.append(col)
for col1 in y_male.columns:
    male_list.append(col1)


# In[69]:


male_df = combined_df.iloc[male_list,:]
female_df = combined_df.iloc[female_list,:]


# In[70]:


male_df_train = male_df.iloc[:180,:]


# In[71]:


#male_df_train


# In[72]:


female_df_train = female_df.iloc[:20,:]


# In[73]:


#female_df_train


# In[74]:


def pcd():
    mw = female_df_train.mean(axis = 0)
    mm = male_df_train.mean(axis = 0)
    means = [mw,mm]
    pcd = pd.concat(means,axis = 1)
    return pcd


# In[75]:


def stds():
    mw = female_df_train.mean(axis = 0)
    mm = male_df_train.mean(axis = 0)
    stdw = np.std(female_df_train,axis = 0)
    stdm = np.std(male_df_train, axis = 0)
    
    stdd = [stdw,stdm]
    stds = pd.concat(stdd,axis=1)
    return stds


# In[76]:


def priors():
    priom = (len(male_df_train) / len(x_train))
    priow = (len(female_df_train) / len(x_train))
    prios = [priom,priow]
    return prios


# In[77]:


priors = priors()


# In[78]:


pcd = pcd()
pcd = pcd.iloc[:-1,:]


# In[79]:


stds = stds()
stds = stds.iloc[:-1,:]


# In[89]:

print("Standard Deviations of Labels (0 indicates women(1),1 indicates men(2))\n")
print(stds)
print("\n")


# In[81]:

print("Sample Means of Labels (0 indicates women(1),1 indicates men(2))\n")

print(pcd)
print("\n")


# In[82]:

print("Prior Probabilites of Men and Women Respectively.\n")
print(priors)
print("\n")


# In[83]:


from math import e
def safelog(val):
    return np.log(val + 1e-100)


# In[84]:


def classify(image):
    log_pcd = pcd.apply(safelog)
    log_one_minus_pcd = (1-pcd).apply(safelog)
    #pcdlerin square bracketlerından birini sildim
    wscore = (np.sum( np.dot((log_pcd[0]) , image)) + np.sum(np.dot(log_one_minus_pcd[0] , (1 - image))) )
    mscore = (np.sum( np.dot((log_pcd[1]) , image)) + np.sum(np.dot(log_one_minus_pcd[1] , (1 - image))) )
    
    
    wscore = np.array(wscore)
    mscore = np.array(mscore)
    

    
    train_scores = [wscore,mscore]
    
    train_scores = np.asarray(train_scores)
    
    #print(train_scores)
    maxed = []
    predicted_final = []
    
    (predicted_final.append(train_scores))
        
    predicted_final = np.asarray(predicted_final)

    return predicted_final


# In[85]:


from sklearn.preprocessing import LabelEncoder
y_encoder = LabelEncoder()
categorized_train_y = y_encoder.fit_transform(y_train)
categorized_test_y = y_encoder.fit_transform(y_test)


# In[86]:


def predict(rows,data):
    classified = []
    y_pred = []
    
    for i in range(0,rows):
        classified.append(classify(data.iloc[i]))
        y_pred.append(np.argmax(classified[i]))
    classified = np.asarray(classified)
    y_pred = np.asarray(y_pred)
    return y_pred


# In[87]:


y_train_pred = predict(rows = len(x_train.index),data = x_train)


# In[88]:


y_test_pred = predict(rows = len(x_test.index),data = x_test ) 


# In[39]:


#y_test_pred


# In[40]:


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(categorized_train_y,y_train_pred)
cm_test = confusion_matrix(categorized_test_y,y_test_pred)


# In[41]:

print("Confusion Matrix of Train Data\n")
print(cm_train)
print("\n")

# In[90]:
print("Confusion Matrix of Test Data\n")
print(cm_test)


# In[ ]:




