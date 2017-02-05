
# for finding various errors by cross validation

def crossValidation(model,training_features,training_labels) :
    n = len(training_labels)
    #print n
    import math
    import numpy
    meanAbsoluteError = 0

    for k in range(0, n):
        tempTF = []
        tempTL = []

        tempTestTF = []
        tempTestTL = []

        for i in range(0, n):
            if i != k:
                tempTF.append(training_features[i])
                tempTL.append(training_labels[i])
            else:
                tempTestTF.append(training_features[i])
                tempTestTL.append(training_labels[i])

        tempTF = numpy.vstack(tempTF)
        tempTL = numpy.array(tempTL)
        tempTestTF = numpy.vstack(tempTestTF)
        tempTestTL = numpy.array(tempTestTL)

        model.fit(tempTF, tempTL)
        x = model.predict(tempTestTF)
        meanAbsoluteError = meanAbsoluteError + math.fabs(x - tempTestTL)
        print "ITERATOR (Actual , Predicted ) --> " ,k,": ( ",x ,", ", tempTestTL,") "
        # print "Mean Absolute Error %d " , meanAbsoluteError

        # print len(tempTestTF)
    meanAbsoluteError = meanAbsoluteError / n
    return meanAbsoluteError