import numpy as np
import pandas as pd
import random

# lr=1e-7, c=1e6, average_validation_accuracy=0.8271
# Epoch=389 train_accuracy=0.8930 test_accuracy=0.8556 (surprisely nice looking!!!)
random.seed(43)
np.random.seed(43)
DATA_DIR, LARGE_NUM, TRAIN_EPOCH = '../data/tfidf_misc/', 1e26, 389
w0, lr, c = np.random.uniform(-0.01, 0.01, 10268), 1e-7, 1e6

def round_train(w, lre, C, datarow):
    y, x = datarow[0], np.append(datarow[1:], 1.0)
    if y == 0:
        y = -1
    
    if y * np.dot(w, x) <= 1:
        w = (1-lre)*w + lre*C*y*x
    else:
        w = (1-lre)*w
    w = np.clip(w, -LARGE_NUM, LARGE_NUM)
    return w

def train(w, lr, C, trainset):
    for epoch in range(TRAIN_EPOCH):
        lre = lr/(1+epoch)
        np.random.shuffle(trainset)
        # Training
        for datarow in trainset:
            w = round_train(w, lre, C, datarow)
    
    return w


trainset = pd.read_csv(DATA_DIR + 'tfidf.misc.train.csv', header=0).to_numpy()
w = train(w0, lr, c, trainset)

print("example_id,label")
evalset = pd.read_csv(DATA_DIR + 'tfidf.misc.eval.csv', header=0).to_numpy()
index = 0
for datarow in evalset:
    label, x = 1, np.append(datarow[1:], 1.0)
    if np.dot(w, x) <= 0:
        label = 0
    print("%d,%d" % (index, label))
    index += 1