# for finding various errors by cross validation

def cross_validation(model, training_features, training_labels, folds):
    n = len(training_labels)
    # print n
    import math
    import numpy
    sqMeanError = 0
    mas = 0
    totalpred25 = 0
    totalpred30 = 0
    total_mmre = 0
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
        print x, tempTestTL
        # meanAbsoluteError = [math.fabs(a-b)/b for a, b in zip(x, tempTestTL)]
        meanAbsoluteError = []

        if len(tempTestTL) > 1:
            meanAbsoluteError = [math.fabs(a - b) / b for a, b in zip(x, tempTestTL)]
        else:
            meanAbsoluteError.append(math.fabs(x - tempTestTL[0]) / tempTestTL[0])
            sqMeanError += (x - tempTestTL[0]) ** 2
            i += 1
        print meanAbsoluteError

        mas = sum(meanAbsoluteError)
        mas /= len(tempTestTL)
        print mas
        if mas <= 0.25:
            totalpred25 += 1.0
        if mas <= 0.30:
            totalpred30 += 1.0
        total_mmre += mas
        # print  total_mmre
        print "ITERATOR (Actual , Predicted ) --> ", k, ": ( ", x, ", ", tempTestTL, ") "
        # print "Mean Absolute Error %d " , meanAbsoluteError
        # print len(tempTestTF)
        counter = counter + 1

    print " TOTAL MMRE ", total_mmre / counter
    print " Squared Mean Error ", sqMeanError / counter
    print " Pred 25 Error ", totalpred25 / counter
    print " Pred 30 Error ", totalpred30 / counter

    return total_mmre
