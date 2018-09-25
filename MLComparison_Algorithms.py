#----------- MAchine learning algorithms for performance comparison ----------------
"""
@author: Shubha Mishra, Prateek Shrivastava
"""
import time
start_time = time.time()

import numpy as np  
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# read data
dataframe = pd.read_csv('Hbdata.csv')

# create input X_values and y_values
X_values = dataframe.ix[:,['age','year','nodes','status']]
y_values = dataframe['status']

le = preprocessing.LabelEncoder()
X = np.array(dataframe)
y = np.array(dataframe)[:,3]
for i in range(0,3):
     X[:,i] = le.fit_transform(X[:,i])

y[:] = le.fit_transform(y[:])

# Split the data as training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.7, random_state=0)

#3 Classification using Linear SVM
from sklearn.svm import SVC
svc_l = SVC(kernel="linear", C=0.025)
svc_l = svc_l.fit(X_train,y_train)
prediction = svc_l.predict(X_test)
#print("3. Linear SVM ", prediction)
print(accuracy_score(y_test, prediction))

#4 Classification using RBF SVM  
from sklearn.svm import SVC
svc_rbf = SVC(gamma=1, C=2)
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print(accuracy_score(y_test, prediction))

#5 Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth= 1, n_estimators=10, max_features=1)
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
#print("5. RandomForestClassifier", prediction)
print(accuracy_score(y_test, prediction))

#8 Classification using logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print(accuracy_score(y_test, prediction))

print("--- %s seconds ---" % (time.time() - start_time))

