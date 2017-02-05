import random
import numpy as numpy
import pandas as pd
import math


def view_model(model):
    """
    Look at model coeffiecients
    """
    print model.coef_, model.intercept_


input_file = "QUES.csv"
input_file2 = "UIMS1.csv"

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

x = numpy_array[:, 0:10]
y = numpy_array[:, 10]

from sklearn.linear_model import LinearRegression

clf = LinearRegression()


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

'''
training_features = X
training_labels = Y
#training_features = [x for (y,x) in sorted(zip(Y,X), key=lambda pair: pair[0])]
#training_labels = [y for (y,x) in sorted(zip(Y,X), key=lambda pair: pair[0])]
<<<<<<< HEAD
k=0;
for i in training_labels :
    print training_features[k] , training_labels[k]
    k=k+1

import tensorflow as tf
||||||| merged common ancestors
k=0;
for i in training_labels :
    print training_features[k] , training_labels[k]
    k=k+1
import tensorflow as tf
=======
#k=0;
#for i in training_labels :
#    print training_features[k] , training_labels[k]
#    k=k+1
>>>>>>> 1ab188b154682d57acc373822c20bcc1f070dd4a
#print (training_features)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

clf = LinearRegression()
clf.fit(train_in, train_out)
sc = clf.predict(test_in)
sc_up = (test_out - sc)/71
err = sum(sc_up)
print err


sc = clf.predict(training_features)
#print "printing scores" , sc
import math
k=0;
for i in sc :
    #print training_labels[k] - sc[k]
    #sc[k] = (i-trainii-training_labels[k]ng_labels[k])**2
    sc[k] = math.pow(i-training_labels[k], 2)
    print k ,  sc[k]
    k=k+1;


from sklearn.metrics import r2_score,mean_squared_error
print r2_score(training_labels,sc)

print mean_squared_error(training_labels, sc)
#print sc
#print training_labels
view_model(clf)

##


index = []

from outlierFunction import outliers

clearedData = outliers(training_features,training_labels,sc)
training_features = []
training_features, training_labels, sc,index = zip(*clearedData)
training_features = numpy.asarray(training_features)
training_labels = numpy.asarray(training_labels)
index= numpy.asarray(index)
print sc


k=0;
for i in sc :
    # print training_labels[k] - sc[k]
    # sc[k] = (i-training_labels[k])**2
    # sc[k] = math.pow(i-training_labels[k], 2)
    print index[k] ,  sc[k]
    k=k+1;

clf.fit (training_features , training_labels)
<<<<<<< HEAD

scores = cross_val_score(clf, training_features, training_labels,cv=10,scoring='r2')

print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
'''
scores = cross_val_score(clf, training_features, training_labels,cv=10,scoring='r2')
print (scores)
print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
=======
'''


#scores = cross_val_score(clf, training_features, training_labels,cv=10,scoring='neg_mean_squared_error')
#print (scores)
#print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from CrossValidation import crossValidation
print crossValidation(clf,training_features,training_labels)
>>>>>>> 1ab188b154682d57acc373822c20bcc1f070dd4a
