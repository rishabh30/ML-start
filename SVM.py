

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import pandas as pd

input_file = "QUES.csv"
input_file2 ="UIMS1.csv"

# comma delimited is the default
df = pd.read_csv(input_file, header=0)
df2 = pd.read_csv(input_file2, header=0)
# for space delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = " ")

# for tab delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = "\t")

# put the original column names in a python list
original_headers = list(df.columns.values)
original_headers2 = list(df2.columns.values)
# print (original_headers)
# remove the non-numeric columns
df = df._get_numeric_data()
df2 = df2._get_numeric_data()
# put the numeric column names in a python list
numeric_headers = list(df.columns.values)
numeric_headers2 = list(df.columns.values)
# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()
numpy_array2 = df2.as_matrix()

# numpy_array = np.concatenate((numpy_array,numpy_array2))

# print (numpy_array)

'''
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(numpy_array)
X_train_minmax
'''

X = numpy_array[:,0:10]
y = numpy_array[:,10]


# np = np.asarray(numpy_array, dtype=float)

###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=9950, gamma=0.0000003)
y_rbf = svr_rbf.fit(X, y)

###############################################################################

from sklearn.model_selection import cross_val_score
scores = cross_val_score(svr_rbf, X,y,scoring='r2')
print (scores)
print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))