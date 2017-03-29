# for finding various errors by cross validation

def cross_validation(model, training_features, training_labels, folds, SQNO):
    n = len(training_labels)
    # print n
    import numpy

    mas = 0
    total_mmre = 0
    counter = 0
    increment = n / folds
    print increment

    clearedData = []

    print training_features
    print training_labels

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
        model.fit([tempTF, tempTF, tempTF], tempTL)
        x = model.predict([tempTestTF, tempTestTF, tempTestTF])
        print x, tempTestTL
        # meanAbsoluteError = [math.fabs(a-b)/b for a, b in zip(x, tempTestTL)]
        meanAbsoluteError = []

        if (len(tempTestTL) > 1):
            meanAbsoluteError = [(a - b) ** 2 for a, b in zip(x, tempTestTL)]
        else:
            meanAbsoluteError.append((x - tempTestTL[0]) ** 2)
            i = i + 1

        mas = sum(meanAbsoluteError)
        mas = mas / len(tempTestTL)
        total_mmre += mas
        counter = counter + 1
        clearedData.append((tempTestTF[0], tempTestTL[0], meanAbsoluteError));

    total_mmre /= counter
    clearedData.sort(key=lambda val: val[2])
    print "TOTAL MMRE : ", SQNO, " : ", total_mmre
    # print  clearedData
    print "::::::::::::::::::::::::::::"
    clearedData.pop()
    # print clearedData

    return clearedData
