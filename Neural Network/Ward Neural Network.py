import copy

import numpy
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import BaseWrapper
from keras.wrappers.scikit_learn import KerasRegressor


def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res


BaseWrapper.get_params = custom_get_params


# define base mode
def baseline_model():
    # create model
    from keras.layers import Merge

    left_branch = Sequential()
    left_branch.add(Dense(3, input_dim=3, activation='tanh'))
    # left_branch.add(Dense(1))
    right_branch = Sequential()
    right_branch.add(Dense(3, input_dim=3, activation='sigmoid'))

    third = Sequential()
    third.add(Dense(3, input_dim=3, init='normal', activation='relu'))

    merged = Merge([left_branch, right_branch, third], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(1))

    final_model.compile(loss='mean_squared_error', optimizer='adam')
    return final_model


input_file = "QUES.csv"
input_file2 = "UIMS1.csv"

df = pd.read_csv(input_file, header=0)
df2 = pd.read_csv(input_file2, header=0)

original_headers = list(df.columns.values)
original_headers2 = list(df2.columns.values)

df = df._get_numeric_data()
df2 = df2._get_numeric_data()

numeric_headers = list(df.columns.values)
numeric_headers2 = list(df.columns.values)

numpy_array = df.as_matrix()
numpy_array2 = df2.as_matrix()


X = numpy_array[:, 0:10]
Y = numpy_array[:, 10]

# training_features = [x for (y, x) in sorted(zip(Y, X), key=lambda pair: pair[0])]
# training_labels = [y for (y, x) in sorted(zip(Y, X), key=lambda pair: pair[0])]
# training_features = numpy.vstack(training_features)
# training_labels = numpy.array(training_labels)
training_features = X
training_labels = Y


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
training_features = SelectKBest(chi2, k=3).fit_transform(training_features, training_labels)



# training_features = [x for (y, x) in sorted(zip(Y, X), key=lambda pair: pair[0])]
# training_labels = [y for (y, x) in sorted(zip(Y, X), key=lambda pair: pair[0])]
# training_features = numpy.vstack(training_features)
# training_labels = numpy.array(training_labels)

seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1100, batch_size=5, verbose=0)
'''IndexError: list assignment index out of range
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''

from CrossValidationsquareErrorWard import cross_validation

err = cross_validation(estimator, training_features, training_labels, 71)
print "MMRE ERROR:", err
