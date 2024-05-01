import numpy as np
import pandas as pd

DATA_DIR = 'data/'
#CV_EPOCH, TRAIN_EPOCH = 100, 200
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 2)
b0 = np.random.uniform(-0.01, 0.01)

def train(epoch_num, trainset, lr, u):
    w, b = w0, b0
    for epoch in range(epoch_num):
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            if y*(np.dot(x, w)+b) <= u:
                print("update")
                w += lr*y*x/(epoch+1)
                b += lr*y/(epoch+1)
    return (w, b)

def test(w, b, testset):
    testdata_size, mistake = len(testset), 0
    for datarow in testset:
        y, x = datarow[0], datarow[1:]
        if y * (np.dot(x, w) + b) <= 0:
            mistake += 1
    return float(testdata_size - mistake) / testdata_size


# Training
trainset = pd.read_csv(DATA_DIR + 'train.csv', header=0).to_numpy()
lr, u = 0.001, 0.001
w, b = train(10, trainset, lr, u)
# Testing
testset = pd.read_csv(DATA_DIR + 'train.csv', header=0).to_numpy()
print("Test accuracy =", test(w, b, testset))

