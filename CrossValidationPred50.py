# for finding various errors by cross validation

def cross_validation(model, training_features, training_labels, folds):
    n = len(training_labels)
    # print n
    import math
    import numpy

    mas = 0
    total = 0
    counter = 0
    increment = n / folds
    print increment
    for k in xrange(0, n, increment):
        tempTF = []
        tempTL = []
        tempTestTF = []
        tempTestTL = []

        for i in range(0, n):
            if k <= i < (k + increment):
                tempTestTF.append(training_features[i])
                tempTestTL.append(training_labels[i])
            else:
                tempTF.append(training_features[i])
                tempTL.append(training_labels[i])

        tempTF = numpy.vstack(tempTF)
        tempTL = numpy.array(tempTL)
        tempTestTF = numpy.vstack(tempTestTF)
        tempTestTL = numpy.array(tempTestTL)
        print counter, ":len of training and test data:", len(tempTL), len(tempTestTL)
        if (len(tempTestTL) < increment):
            continue
        model.fit(tempTF, tempTL)
        x = model.predict(tempTestTF)
        # print x, tempTestTL
        # meanAbsoluteError = [math.fabs(a-b)/b for a, b in zip(x, tempTestTL)]
        meanAbsoluteError = []

        if (len(tempTestTL) > 1):
            meanAbsoluteError = [math.fabs(a - b) / b for a, b in zip(x, tempTestTL)]
        else:
            meanAbsoluteError.append(math.fabs(x - tempTestTL[0]) / tempTestTL[0])
            i = i + 1
        print meanAbsoluteError

        mas = sum(meanAbsoluteError)
        mas = mas / len(tempTestTL)

        print mas
        if mas <= 0.30:
            total += 1.0

        # print  total_mmre
        # print "ITERATOR (Actual , Predicted ) --> ", k, ": ( ", x, ", ", tempTestTL, ") "
        # print "Mean Absolute Error %d " , meanAbsoluteError
        # print len(tempTestTF)
        counter = counter + 1
    print total
    total /= counter
    return total
