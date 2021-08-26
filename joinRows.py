import numpy as np
# load
# for i in range(0,25):
#     results = np.genfromtxt('wearable_results_().csv', delimiter=',')

# load
# results_0 = np.genfromtxt('wearable_results_0.csv', delimiter=',')
# print(results_0.shape)
results_0 = np.genfromtxt('prediction.results-00000-of-00023.csv', delimiter=',')
results_1 = np.genfromtxt('prediction.results-00001-of-00023.csv', delimiter=',')
results_2 = np.genfromtxt('prediction.results-00002-of-00023.csv', delimiter=',')
results_3 = np.genfromtxt('prediction.results-00003-of-00023.csv', delimiter=',')
results_4 = np.genfromtxt('prediction.results-00004-of-00023.csv', delimiter=',')
results_5 = np.genfromtxt('prediction.results-00005-of-00023.csv', delimiter=',')
results_6 = np.genfromtxt('prediction.results-00006-of-00023.csv', delimiter=',')
results_7 = np.genfromtxt('prediction.results-00007-of-00023.csv', delimiter=',')
results_8 = np.genfromtxt('prediction.results-00008-of-00023.csv', delimiter=',')
results_9 = np.genfromtxt('prediction.results-00009-of-00023.csv', delimiter=',')
results_10 = np.genfromtxt('prediction.results-00010-of-00023.csv', delimiter=',')
results_11 = np.genfromtxt('prediction.results-00011-of-00023.csv', delimiter=',')
results_12 = np.genfromtxt('prediction.results-00012-of-00023.csv', delimiter=',')
results_13 = np.genfromtxt('prediction.results-00013-of-00023.csv', delimiter=',')
results_14 = np.genfromtxt('prediction.results-00014-of-00023.csv', delimiter=',')
results_15 = np.genfromtxt('prediction.results-00015-of-00023.csv', delimiter=',')
results_16 = np.genfromtxt('prediction.results-00016-of-00023.csv', delimiter=',')
results_17 = np.genfromtxt('prediction.results-00017-of-00023.csv', delimiter=',')
results_18 = np.genfromtxt('prediction.results-00018-of-00023.csv', delimiter=',')
results_19 = np.genfromtxt('prediction.results-00019-of-00023.csv', delimiter=',')
results_20 = np.genfromtxt('prediction.results-00020-of-00023.csv', delimiter=',')
results_21 = np.genfromtxt('prediction.results-00021-of-00023.csv', delimiter=',')
results_22 = np.genfromtxt('prediction.results-00022-of-00023.csv', delimiter=',')
# results_23 = np.genfromtxt('prediction.results-00023-of-00023.csv', delimiter=',')


# concat
args = (results_0, results_1, results_2, results_3, results_4, results_5, results_6, results_7, results_8, results_9,
        results_10, results_11, results_12, results_13, results_14, results_15, results_16, results_17, results_18,
        results_19, results_20, results_21, results_22)
results = np.concatenate(args, axis=0)
print(results.shape)

# sort
sorted_array = results[np.argsort(results[:, 0])]
unique = np.unique(sorted_array, axis=0)

# append correct labels
# labels = np.genfromtxt('wearable_test_0_ans.csv', delimiter=',')
# write
np.savetxt('wearable_results_225_final.csv', unique, delimiter=',')







