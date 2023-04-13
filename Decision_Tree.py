#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#loading data file
data=pd.read_csv('DT.csv',sep=',',header=0)
data.head()


# In[5]:


print("database length:",len(data))


# In[12]:


# seperating the target variable
X=data.values[:,1:5]
Y=data.values[:,0]

# splitting dataset into train and test 
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)

# function to perform training with entropy
clf=DecisionTreeClassifier(criterion="entropy",random_state=100,
                                  max_depth=3,min_samples_leaf=5)
clf.fit(X_train,y_train)


# In[10]:


#function to predict
y_pred=clf.entropy.predict(X_test)
y_pred


# In[13]:


#checking accuracy
print('Accuracy is',accuracy_score(y_test,y_pred)*100)

