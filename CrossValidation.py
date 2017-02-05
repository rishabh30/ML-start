
# for finding various errors by cross validation


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
