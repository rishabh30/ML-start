# for selecting Outliers from the data

def outliers(training_features, training_labels, sc):
    clearedData = []
    k = 0
    for tf, tl, pr in zip(training_features, training_labels, sc):
        clearedData.append((tf, tl, pr, k));
        k = k + 1;
    clearedData.sort(key=lambda val: val[2])
    clearedData = clearedData[0:71]

    return clearedData


'''


    training_features = []
    training_features, training_labels, sc = zip(*clearedData)
    training_features= numpy.asarray(training_features)
    training_labels= numpy.asarray(training_labels)

    print training_features
    print "\n\n\n\n\n"
    print training_labels
    print sc


'''
