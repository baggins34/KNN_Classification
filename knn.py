"""
K Nearest Neighbor Classification
by
Ibrahim Halil Bayat, PhD
Istanbul Technical University
Istanbul, Turkey
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
import matplotlib.ticker as ticker

df = pd.read_csv("teleCust1000t.csv")
print(df['custcat'].value_counts())

"""
1- Basic Services
2- E-service Customers
3- Plus Services 
4- Total Services 
"""
df['income'].hist(bins = 50)
plt.show()

x = df[['region', 'tenure','age', 'marital', 'address', 'income'
        , 'ed', 'employ','retire', 'gender', 'reside']] .values.astype(float)
y = df['custcat'].values.astype(float)
print(x[0:5])
print("**************************************************************************")
# Let's normalize the data and see what changes.
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
print(x[0:5])

print("**************************************************************************")
# Time for train and test split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=4)
print('Train set: ', xtrain.shape, ytrain.shape)
print('Test set: ', xtest.shape, ytest.shape)

# Now it is time to have KNN Classification

from sklearn.neighbors import KNeighborsClassifier

k = 4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(xtrain, ytrain)
print(neigh)

yhat = neigh.predict(xtest)
print(yhat[0:5])

# Accuracy

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(ytrain, neigh.predict(xtrain)))
print("Test set Accuracy: ", metrics.accuracy_score(ytest, yhat))