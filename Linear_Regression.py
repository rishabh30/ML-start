import pandas as pd
import array


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

X = numpy_array[:, 0:10]
Y = numpy_array[:, 10]

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

training_features = X
training_labels = Y

#using PCA to reduce 10 components to 3 components
k = 0
eg1 = [-0.0701, 0.00, -0.0449, -0.389, -0.3596, -0.3551, -0.3516, -0.4056, -0.4083, -0.3626]
eg2 = [-0.4532, 0.00, 0.6624, 0.1768, -0.1646, -0.2196, 0.2676, -0.1697, -0.1884, 0.3367]
eg3 = [0.7561, 0.00, 0.523, 0.0794, -0.1738, 0.224, -0.2009, -0.1441, -0.0489, 0.0678]

for i in training_labels:
    count = 0;
    pc1 = 0.0;
    pc2 = 0.0;
    pc3 = 0.0;
    for j in training_features[k]:
        pc1 += (eg1[count]) * (training_features[k][count])
        pc2 += (eg2[count]) * (training_features[k][count])
        pc3 += (eg3[count]) * (training_features[k][count])
        count += 1

    training_features[k][0] = pc1
    training_features[k][1] = pc2
    training_features[k][2] = pc3
    k += 1

training_features_new = training_features[:, 0:3]

# Without PCA and featureSelection using 10 components

'''

#training_features = [x for (y,x) in sorted(zip(Y,X), key=lambda pair: pair[0])]
#training_labels = [y for (y,x) in sorted(zip(Y,X), key=lambda pair: pair[0])]

k=0;
for i in training_labels :
    print training_features[k] , training_labels[k]
    k=k+1

k=0;
for i in training_labels :
    print training_features[k] , training_labels[k]
    k=k+1

#k=0;
#for i in training_labels :
#    print training_features[k] , training_labels[k]
#    k=k+1
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
'''

#Implementing outlierFunction

'''

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

scores = cross_val_score(clf, training_features, training_labels,cv=10,scoring='r2')

print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''

from CrossValidationMaxMRE import cross_validation

err = cross_validation(clf, training_features_new, training_labels, 71)

print "MMRE ERROR:", err, '\n'

#Using Feature Selection
'''

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE

selector = RFE(clf, 3, step=1)
err2 = cross_validation(selector, training_features, training_labels, 71)
print selector.fit_transform(X, Y)
print "MMRE ERROR AFTER FS", err2
'''

from CrossValidationPred25 import cross_validation

err = cross_validation(clf,training_features,training_labels,71)
print "PRED 25:", err


from CrossValidationPred50 import cross_validation

err = cross_validation(clf,training_features,training_labels,71)
print "PRED 30:", err
