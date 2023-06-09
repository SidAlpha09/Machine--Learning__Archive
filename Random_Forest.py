
#loading the library with the iris flower dataset
from sklearn.datasets import load_iris

#loading scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import random 
#setting random seed
np.random.seed(0)


iris=load_iris()
data_frame=pd.DataFrame(iris.data,columns=iris.feature_names)
data_frame.head()


data_frame['species']=pd.Categorical.from_codes(iris.target,
                                               iris.target_names)
data_frame.head()



#creating test and training data
data_frame['is_train']=np.random.uniform(0,1,len(data_frame))<=.75
data_frame.head()



#creating dataframes with test rows and training rows
train,test=data_frame[data_frame['is_train']==True],data_frame[data_frame['is_train']==False]

#printing the observations
print('no. of observations in training data:',len(train))
print('no. of observations in testing data:',len(test))




#creating a list of features column's name
features=data_frame.columns[:4]
features



#converting each species name into digits
arr=pd.factorize(train['species'])[0]

arr



clf=RandomForestClassifier(n_jobs=2,random_state=0)

#training classifier
clf.fit(train[features],arr) #taking training set features and target is y




#applying the trained classifier to the test
clf.predict(test[features])




#viewing the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]




#mapping names for the plants for each predicted plant class
preds=iris.target_names[clf.predict(test[features])]

#viewing the predicted species for the first five observations
preds[10:15]




# viewing actual species for the first five observations
test['species'].head()




#creating confusion matrix which combines our predictions and make it in a single matrix
pd.crosstab(test['species'],preds,rownames=['Actual Species'],
           colnames=['Predicted Species'])


# it will give 93% accuracy (30/32)*100




# end product which is gonna be deployed
preds=iris.target_names[clf.predict([[5.0,3.6,1.4,2.0],[5.0,3.6,1.4,2.0]])]
preds

