import pandas as pd


def view_model(model):
    """
    Look at model coeffiecients
    """
    print "\n Model coeffiecients"
    print "======================\n"
    for i, coef in enumerate(model.coef_):
        print "\tCoefficient %d  %0.3f" % (i + 1, coef)

    print "\n\tIntercept %0.3f" % (model.intercept_)


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



training_features = numpy_array[:, 0:10]
training_labels = numpy_array[:, 10]

# print (training_features)

from sklearn import linear_model

clf = linear_model.Lasso(alpha=9)

clf.fit(training_features, training_labels)
sc = clf.predict(training_features)
from sklearn.metrics import r2_score, mean_squared_error

print r2_score(training_labels, sc)
print mean_squared_error(training_labels, sc)
# print sc
# print training_labels
view_model(clf)
##

from sklearn.model_selection import cross_val_score

# clf.fit (training_features , training_labels)
scores = cross_val_score(clf, training_features, training_labels, cv=6)
print (scores)
print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
'''
