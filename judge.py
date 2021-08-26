import csv
from sklearn.metrics import jaccard_score
import os

import csv
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import os
import numpy as np

# train = np.loadtxt("train.csv", delimiter=',', encoding='utf-8-sig')
# print(train.shape)


if __name__ == '__main__':
    # os.chdir('pca_200')
    results = 'wearable_results_225_final.csv'
    # test = 'movie_PCA_test_{}.csv'
    testLabels = 'wearable_test_y.csv'
    globalLabels = []
    globalPredictions = []
    for i in range(0, 1):
        f1 = open(results, encoding='utf-8-sig')
        # f2 = open(test.format(i))
        f3 = open(testLabels, encoding='utf-8-sig')

        r1 = csv.reader(f1)
        # r2 = csv.reader(f2)
        r3 = csv.reader(f3)
        labels = []
        predictions = []

        # next(r1)
        for row in r1:
            # prediction = int(float(row[-7]))
            row = [int(float(element)+0.5) for element in row]
            classes = row[-6:]
            maxClass = max(classes)
            classesIndex = classes.index(maxClass)
            myClass = classesIndex
            predictions.append(myClass)
        # next(r3)
        for row in r3:
            label = int(row[-1])
            labels.append(label)
        globalLabels.extend(labels)
        globalPredictions.extend(predictions)

        assert len(predictions) == len(labels)

        n_samples = [0, 0, 0, 0, 0, 0]
        n_correct = [0, 0, 0, 0, 0, 0]
        count = len(predictions)
        for j in range(count):
            p = predictions[j]
            L = labels[j]
            n_samples[L] += 1
            if p == L:
                n_correct[p] += 1

        print(n_correct, n_samples)
        print(round(n_correct[0] / n_samples[0], 3), round(n_correct[1] / n_samples[1], 3))

        scores = jaccard_score(labels, predictions, labels=[0, 1, 2, 3, 4, 5], average=None)
        print('jaccard indices = {}'.format(scores))

        precision = precision_score(labels, predictions, labels=[0, 1, 2, 3, 4, 5], average=None)
        print('precision: {}'.format(precision))

        recall = recall_score(labels, predictions, labels=[0, 1, 2, 3, 4, 5], average=None)
        print('recall: {}'.format(recall))

        accuracy = accuracy_score(labels, predictions)
        print('accuracy: {}'.format(accuracy))


        f1.close()
        f3.close()
    print('Global Jaccard Indices:')
    print(jaccard_score(globalLabels, globalPredictions, labels=[0, 1, 2, 3, 4, 5], average=None))

