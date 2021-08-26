import csv
from sklearn.metrics import jaccard_score
import os
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from sklearn.impute import KNNImputer

import pandas as pd

# train = np.loadtxt("X_train.txt", delimiter=' ', encoding='utf-8-sig')
# print(train.shape)
# test = np.loadtxt("X_test.txt", delimiter=' ', encoding='utf-8-sig')
# print(train.shape)
# train2 = np.loadtxt("wearable_pca225_train_X.csv", delimiter=',', encoding='utf-8-sig')
# print(train2.shape)
# test2 = np.loadtxt("wearable_pca225_test_X.csv", delimiter=',', encoding='utf-8-sig')
# print(test2.shape)

if __name__ == '__main__':
    for i in range(0, 1):
        # get data
        train = np.loadtxt("X_train.csv", delimiter=',', encoding='utf-8-sig')
        # train = np.loadtxt("train_0.csv", delimiter=',', encoding='utf-8-sig')
        # trainLabels = train[:, -1]
        trainLabels = np.loadtxt("wearable_train_y.csv", delimiter=',', encoding='utf-8-sig')
        print(train.shape)
        # train = train[:, :-1]
        test = np.loadtxt("X_test.csv", delimiter=',', encoding='utf-8-sig')
        # test = np.loadtxt("test_0.csv", delimiter=',', encoding='utf-8-sig')
        # testLabels = test[:, -1]
        print(test.shape)
        # test = test[:, :-1]
        # print(testLabels.shape)
        # test = read_csv("SelectQuoteAgentDealsMatched.csv", header=None, na_values='?')
        # print("test")
        # imputer = KNNImputer()
        # imputer.fit(testData)
        # testTrans = imputer.transform(testData)


        # PCA
        start = time.time()
        pca = PCA(n_components=172)

        # normalize
        sc = StandardScaler()
        normTrain = sc.fit(train)
        train_t = sc.transform(train)
        test_t = sc.transform(test)
        # file_name = 'scaled.csv'
        # np.savetxt(file_name, train_t, delimiter=',')
        print(i)
        # fit
        pca.fit(train_t)
        # print(pca.components_.shape)
        train_f = pca.transform(train_t)
        # print(train_f)
        test_f = pca.transform(test_t)
        # print(test_f)

        # clock and explained variance
        print('Duration: {} seconds'.format(time.time() - start))
        # variance
        # print(pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_))
        trainLabels = trainLabels.reshape(trainLabels.shape[0], -1)
        #
        # # replicate (weight class 1)
        # num_repeats = 11
        # my_new_data = []
        train_final = np.concatenate((train_f, trainLabels), axis=1)
        # print(train_final.shape)
        # for row in train_final:
        #     current_row = np.array(row)
        #     classes = int(current_row[-1])
        #     if classes == 1:
        #         for x in range(num_repeats):
        #             my_new_data.append(row)
        #     else:
        #         my_new_data.append(row)
        #
        # write reduced data
        # training
        file_name = 'wearable_pca172_train.csv'.format(i)
        np.savetxt(file_name, train_final, delimiter=',')
        print(train_f.shape)
        # testing
        # testLabels = testLabels.reshape(testLabels.shape[0], -1)
        file_name = 'wearable_pca172_test.csv'.format(i)
        np.savetxt(file_name, test_f, delimiter=',')
        # file_name = 'PCA170_testLabels_{}.csv'.format(i)
        # np.savetxt(file_name, testLabels, delimiter=',')

