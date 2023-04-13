#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import numpy as np
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
digits=load_digits()


# In[2]:


#determining total number of images and labels
print('Image Data shape',digits.data.shape)
print('Label Data shape',digits.target.shape)


# In[3]:


# displaying some of the images and labels
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n'%label,fontsize=20)


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[10]:


# making an instance of the model and training it
model=LogisticRegression()
model.fit(x_train,y_train)


# In[11]:


# predicting the output of the first element of the test set
print(model.predict(x_test[0].reshape(1,-1)))


# In[12]:


# predicting the output of the first 10 elements of the set 
model.predict(x_test[0:10])


# In[14]:


#predicting entire dataset
prediction=model.predict(x_test)
score=model.score(x_test,y_test)
print(score)


# In[18]:


cm=metrics.confusion_matrix(y_test,prediction) #the more the number in the diagonal the better the accuracy is

print(cm)


# In[22]:


# representing the confusion matrix ina heat map

plt.figure(figsize=(0,9))
sns.heatmap(cm,annot=True,fmt=".3f",linewidths=5,square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title='Accuracy Score: {0}'.format(score)
plt.title(all_sample_title,size=15)


# In[ ]:




