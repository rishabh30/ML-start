

import numpy as np
from sklearn.svm import SVR
from sklearn import tree
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
Y = numpy_array[:,10]


clf = tree.DecisionTreeRegressor()


###############################################################################
##R2 SCORE
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, Y, scoring='r2',cv=10)
print (scores)
print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

##MMRE SCORE
def cross_validation(model, training_features, training_labels, folds):
    n = len(training_labels)
    # print n
    import math
    import numpy

    mas = 0

    for k in xrange(folds, n, folds):
        tempTF = []
        tempTL = []
        tempTestTF = []
        tempTestTL = []

        for i in range(0, n):
            if k > i >= (k - folds):
                tempTestTF.append(training_features[i])
                tempTestTL.append(training_labels[i])
            else:
                tempTF.append(training_features[i])
                tempTL.append(training_labels[i])

        tempTF = numpy.vstack(tempTF)
        tempTL = numpy.array(tempTL)
        tempTestTF = numpy.vstack(tempTestTF)
        tempTestTL = numpy.array(tempTestTL)

        model.fit(tempTF, tempTL)
        x = model.predict(tempTestTF)
        meanAbsoluteError = (math.fabs(x - tempTestTL)) / tempTestTL
        mas += sum(meanAbsoluteError)
        print "ITERATOR (Actual , Predicted ) --> ", k, ": ( ", x, ", ", tempTestTL, ") "
        # print "Mean Absolute Error %d " , meanAbsoluteError

        # print len(tempTestTF)

    mas /= n
    return mas


err = cross_validation(clf, x, y, 2)
err1 = err * 71
print "MMRE ERROR:", err
print "MAX MRE:", err1
