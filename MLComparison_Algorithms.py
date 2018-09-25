"""
# Prediction using machine learning algorithms
"""
import numpy as np  
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Read the data
data = pd.read_csv('Hbdata.csv')

# Prepare data for models
y = data['status']
X = data.drop(data.status)

# Split the data as training and testing data - 30% testing, 70% training
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3, random_state = None)

# Classification using Linear SVM
from sklearn.svm import SVC
svc_l = SVC(kernel = "linear", C = 0.025)
svc_l = svc_l.fit(X_train,y_train)
prediction = svc_l.predict(X_test)
#print("3. Linear SVM ", prediction)
print(accuracy_score(y_test, prediction))

# Classification using RBF SVM  
from sklearn.svm import SVC
svc_rbf = SVC(gamma=1, C=2)
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print(accuracy_score(y_test, prediction))

# Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth= 1, n_estimators=10, max_features=1)
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
#print("5. RandomForestClassifier", prediction)
print(accuracy_score(y_test, prediction))

# Classification using Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print(accuracy_score(y_test, prediction))


