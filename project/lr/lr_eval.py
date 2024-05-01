import numpy as np
import pandas as pd
import random
import math

#best_lr=0.500000, best_sigma=10000000.000000
# TRAIN_EPOCH=10, test_accuracy = 0.8507
# Epoch=5 train_accuracy=0.8797 test_accuracy=0.8524
random.seed(43)
np.random.seed(43)
DATA_DIR, TRAIN_EPOCH, w0, LARGE_NUM = '../data/tfidf_misc/', 5, np.random.uniform(-0.01, 0.01, 10268), 1e8
lr, sigma, trainset, evalset = 0.5, 10000000.0, pd.read_csv(DATA_DIR + 'tfidf.misc.train.csv', header=0).to_numpy(), pd.read_csv(DATA_DIR + 'tfidf.misc.eval.csv', header=0).to_numpy()

def sigmoid_func(z):
    if z>=8.0:  return 1.0
    elif z<=-8.0:   return 0
    else:   return 1.0 / (1.0+math.exp(-z))

def round_train(w, lre, sigma, datarow):
    y, x = float(datarow[0]), np.append(datarow[1:], 1.0)
    if y == 0:
        y = -1
    w = (1.0 - 2.0*lre/sigma)*w + y*x*lre*sigmoid_func(-y*np.dot(w,x))
    w = np.clip(w, -LARGE_NUM, LARGE_NUM)
    return w

def train(w, lr, sigma, trainset):
    for epoch in range(TRAIN_EPOCH):
        lre = lr/(1+epoch)
        np.random.shuffle(trainset)
        for datarow in trainset:
            w = round_train(w, lre, sigma, datarow)
    return w

w = train(w0, lr, sigma, trainset)
print("example_id,label")
index = 0
for datarow in evalset:
    label, x = 1, np.append(datarow[1:], 1.0)
    if np.dot(w, x) <= 0:
        label = 0
    print("%d,%d" % (index, label))
    index += 1

