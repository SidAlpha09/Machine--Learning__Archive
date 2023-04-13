
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
#make blob is used to create sample data when we are not sure what our data should be

#we create 40 sample points
x,y=make_blobs(n_samples=40,centers=2,random_state=20)

#fit the model ,don't regularize for illustration purposes
clf=svm.SVC(kernel='linear',C=1)
clf.fit(x,y)

#display the data in graph form
plt.scatter(x[:,0],x[:,1],c=y,s=30,cmap=plt.cm.Paired)
plt.show()


# In[14]:


#predicting the unknown data
Data=[[4,7],[10,6]]
print(clf.predict(Data))

# 4,7 is in the 0 side and 10,6 is in the 1 side



#now to see what is happening behind
clf=svm.SVC(kernel='linear',C=1000)
clf.fit(x,y)
plt.scatter(x[:,0],x[:,1],c=y,s=30,cmap=plt.cm.Paired)

#plot the decision function
ax=plt.gca()
x_lim=ax.get_xlim()
y_lim=ax.get_ylim()

#create grid to evaluate model
xx=np.linspace(x_lim[0],x_lim[1],30)
yy=np.linspace(y_lim[0],y_lim[1],30)
YY,XX=np.meshgrid(yy,xx)
xy=np.vstack([XX.ravel(),YY.ravel()]).T
Z=clf.decision_function(xy).reshape(XX.shape)

#plot decision boundary and margins
ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],
          alpha=0.5,
          linestyle=['--','-','--'])

#plot sv
ax.scatter(clf.support_vectors_[:,0],
          clf.support_vectors_[:,1],s=100,
          linewidth=1,facecolors='none')
plt.show()

