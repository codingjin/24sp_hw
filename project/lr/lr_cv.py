import numpy as np
import pandas as pd
import random
import math

random.seed(43)
np.random.seed(43)
DATA_DIR, CV_EPOCH, LARGE_NUM, w0 = '../data/tfidf_misc/', 10, 1e30, np.random.uniform(-0.01, 0.01, 10268)
dataset = []
for i in range(5):
    dataset.append(pd.read_csv(DATA_DIR + f'CVfolders/fold{i}.csv', header=0).to_numpy())


LR_LIST, SIGMA_LIST = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001], [1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 10.0, 1.0, 1e-1, 1e-2]
best_lr, best_sigma, best_cv_accuracy = 100.0, 1000000000.0, -0.01
#best_lr=0.500000, best_sigma=10000000.000000, best_cv_accuracy=0.8305
def sigmoid_func(z):
    if z>=30.0:  return 1.0
    elif z<=-30.0:   return 0
    else:   return 1.0 / (1.0+math.exp(-z))

def train(epoch_num, trainset, w, lr, sigma):
    for epoch in range(epoch_num):
        lre = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1.0)
            if y == 0:
                y = -1
            # update
            w = (1.0 - 2.0*lre/sigma)*w + y*x*lre*sigmoid_func(-y*np.dot(w,x))
            w = np.clip(w, -LARGE_NUM, LARGE_NUM)

    return w

def test(w, testset):
    size, num_mistake = len(testset), 0
    for datarow in testset:
        y, x = datarow[0], np.append(datarow[1:], 1.0)
        if y == 0:
            y = -1
        
        if y * np.dot(w, x) <= 0:
            num_mistake += 1
    
    return float(size-num_mistake) / size

for lr in LR_LIST:
    sequential_stop_condition = 0
    for sigma in SIGMA_LIST:
        accuracies, stop_condition = [], 0
        for i in range(5):
            validationset = dataset[i]
            trainset = np.concatenate((dataset[(i+1)%5], dataset[(i+2)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+3)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+4)%5]), axis=0)
            
            w = train(CV_EPOCH, trainset, w0, lr, sigma)
            # validation
            temp_accuracy = test(w, validationset)
            if temp_accuracy < 0.7:
                print("lr=%f, sigma=%f are bad hyperparameters! cv_accuracy=%f" % (lr, sigma, temp_accuracy))
                stop_condition = 1
                sequential_stop_condition += 1
                break
            accuracies.append(temp_accuracy)

        if stop_condition == 1:
            if sequential_stop_condition == 2:
                break
            continue
        else:
            sequential_stop_condition = 0

        accuracy = np.mean(accuracies)
        print("lr=%f, sigma=%f, cv_accuracy=%.4f" % (lr, sigma, accuracy))
        if accuracy > best_cv_accuracy:
            best_lr, best_sigma, best_cv_accuracy = lr, sigma, accuracy

print("best_lr=%f, best_sigma=%f, best_cv_accuracy=%.4f" % (best_lr, best_sigma, best_cv_accuracy))
#best_lr=0.500000, best_sigma=10000000.000000, best_cv_accuracy=0.8305
