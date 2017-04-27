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
    left_branch.add(Dense(3, input_dim=3, activation='softplus'))
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
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
training_features_new = SelectKBest(chi2, k=3).fit_transform(training_features, training_labels)
'''
# print X,Y
seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1100, batch_size=5, verbose=0)
'''IndexError: list assignment index out of re(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''



from MaxSquaredError import cross_validation

i = 0
while i < 10:
    print training_labels.size
    clearedData = cross_validation(estimator, training_features_new, training_labels, training_labels.size, i)
    training_features_new, training_labels, sc = zip(*clearedData)
    training_features_new = numpy.asarray(training_features_new)
    training_labels = numpy.asarray(training_labels)
    sc = numpy.asarray(sc)
    i += 1

print "FINAL MMRE ERROR:", sc.mean()

from CrossValidationsquareErrorWard import cross_validationall

err = cross_validationall(estimator, training_features_new, training_labels, 61)
