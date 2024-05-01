import numpy as np
import pandas as pd

DATA_DIR = 'data/'
#CV_EPOCH, TRAIN_EPOCH = 100, 200
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 3)
b0 = np.random.uniform(-0.01, 0.01)

def train(epoch_num, trainset, lr):
    w, b = w0, b0
    for epoch in range(epoch_num):
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            if y*(np.dot(x, w)+b) <= 0:
                w += lr*y*x
                b += lr*y
    return (w, b)

def test(w, b, testset):
    testdata_size, mistake = len(testset), 0
    for datarow in testset:
        y, x = datarow[0], datarow[1:]
        #print(x)
        #print(w)
        if y * (np.dot(x, w) + b) <= 0:
            mistake += 1
    return float(testdata_size - mistake) / testdata_size

def traintest(epoch_num, trainset, lr, testset):
    w, b = w0, b0
    for epoch in range(epoch_num):
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            #print(x)
            #print(w)
            if y*(np.dot(x, w)+b) <= 0:
                w += lr*y*x
                b += lr*y

        print("Epoch %d: trainacc=%.2f" % (epoch, test(w, b, testset)))
        print("w=", w, " b=", b)

# Training
trainset = pd.read_csv(DATA_DIR + 'hw5.csv', header=0).to_numpy()
lr = 0.01
#w, b = train(500, trainset, lr)
# Testing
testset = pd.read_csv(DATA_DIR + 'hw5.csv', header=0).to_numpy()
traintest(100, trainset, lr, testset)
# Testing
#testset = pd.read_csv(DATA_DIR + 'train.csv', header=0).to_numpy()
#print("Test accuracy =", test(w, b, testset))

