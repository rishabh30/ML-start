from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import train_test_split

from sklearn import metrics

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("QUES.csv", delimiter=",")
train, test = train_test_split(dataset, test_size=0.1)

# split into input (X) and output (Y) variables
train_X = train[:, 0:10]
train_Y = train[:, 10]
# create model
model = Sequential()
model.add(Dense(10, input_dim=10, init='normal', activation='softplus'))
model.add(Dense(1, init='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(train_X, train_Y, nb_epoch=20000, batch_size=5, verbose=1)

# calculate predictions
test_X = test[:, 0:10]
actual = test[:, 10]
predictions = model.predict(test_X)
# round predictions
B = []
for x in predictions:
    for i in x:
        #        print(i)
        B.append(i)
print("Actual\t\tPredicted\t MRE\t\tRes. Error\tAbs. Res. Error")
sum, sum2 = 0, 0
max, mre, res, abres = 0, 0, 0, 0
for i, j in zip(actual, B):
    mre = abs(i - j) / i
    res = i - j
    abres = abs(res)
    sum = sum + mre
    sum2 = sum2 + res
    if (abs(i - j) / i > max):
        max = abs(i - j) / i
    stringI = '{:.4f}'.format(i)
    stringJ = '{:.4f}'.format(j)
    print (stringI, "\t", stringJ, "\t", '{:.4f}'.format(mre), "\t", '{:.4f}'.format(res), "\t", '{:.4f}'.format(abres))

print("\nMean Magnitude of Relative Error (MMRE) : ", sum / len(actual))
print("MaxMRE : ", max)
print("Explained Variance Score", metrics.explained_variance_score(actual, B))
print("Mean Absolute Error", metrics.mean_absolute_error(actual, B))
print("Mean Squared Error", metrics.mean_squared_error(actual, B))
print("Median Absolute Error", metrics.median_absolute_error(actual, B))
print("R-Square Score", metrics.r2_score(actual, B))
print("Root Mean Square Error : ", (sum2 / len(B)) ** (1 / 2))