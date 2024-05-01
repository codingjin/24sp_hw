import numpy as np
import pandas as pd

DATA_DIR = 'data/'
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 221)


#trainset = pd.read_csv(DATA_DIR + 'train.csv', header=0).to_numpy()
trainset = pd.read_csv(DATA_DIR + 'debug.csv', header=0).to_numpy()
testset = pd.read_csv(DATA_DIR + 'test.csv', header=0).to_numpy()
"""
row0 = trainset[0]
print(row0)
row00 = np.append(row0, 1)
print(row00)
"""

epoch_num = 10
w = w0
lr = 0.5
C = 1
train_acc, train_len, test_acc, test_len = [], len(trainset), [], len(testset)
for epoch in range(epoch_num):
    lre, train_mistake, test_mistake = lr/(epoch+1), 0, 0
    np.random.shuffle(trainset)
    for datarow in trainset:
        y, x = datarow[0], np.append(datarow[1:], 1)
        if y * np.dot(w, x) <= 1:
            w = (1-lre)*w + lre*C*y*x
            if y * np.dot(w, x) < 0:
                train_mistake += 1
        else:
            w = (1-lre)*w

    for test_datarow in testset:
        y, x = test_datarow[0], np.append(test_datarow[1:], 1)
        if y * np.dot(w, x) < 0:
            test_mistake += 1
    
    train_acc.append(float(train_len-train_mistake) / train_len)
    test_acc.append(float(test_len-test_mistake) / test_len)

for i in range(epoch_num):
    print("Epoch %d, training accuracy=%.2f, test accuracy=%.2f" % (i, train_acc[i], test_acc[i]))






