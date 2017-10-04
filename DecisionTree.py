# USING DecisionTreeRegressor for Prediction
import pandas as pd
from sklearn import tree

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

'''
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(numpy_array)
X_train_minmax
'''

X = numpy_array[:, 0:10]
Y = numpy_array[:, 10]

clf = tree.DecisionTreeRegressor()

###############################################################################
##R2 SCORE
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, Y, scoring='r2', cv=10)
print (scores)
print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

##MMRE SCORE
