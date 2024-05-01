import numpy as np
import pandas as pd

DATA_DIR = 'data/'
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 3)
#w0 = np.zeros(221)
CV_EPOCH = 10


def cv_train(epoch_num, trainset, lr, C):
    w = w0
    for epoch in range(epoch_num):
        lre = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1)
            if y * np.dot(w, x) <= 1:
                w = (1-lre)*w + lre*C*y*x
            else:
                w = (1-lre)*w
    return w

def cv_validation(w, validationset):
    tp, tn, fp, fn = 0, 0, 0, 0
    for datarow in validationset:
        y, x = datarow[0], np.append(datarow[1:], 1)
        if y * np.dot(w, x) < 0:
            if y==1:    fn += 1
            else:       fp += 1
        else:
            if y==1:    tp += 1
            else:       tn += 1
    if tp+fp == 0:  p=1
    else:           p = float(tp)/float(tp+fp)
    r = float(tp)/float(tp+fn)
    f1 = 2*p*r/(p+r)
    return (p, r, f1)

"""
dataset = []
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training00.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training01.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training02.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training03.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training04.csv', header=0).to_numpy())

LR_LIST, C_LIST = [1, 0.1, 0.01, 0.001, 0.0001], [10, 1, 0.1, 0.01, 0.001, 0.0001]
best_lr, best_c, best_p, best_r, best_f1 = 1, 10, 0.0, 0.0, -0.01
for lr in LR_LIST:
    for c in C_LIST:
        validation_p, validation_r, validation_f1 = [], [], []
        for i in range(5):
            validationset = dataset[i]
            trainset = np.concatenate((dataset[(i+1)%5], dataset[(i+2)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+3)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+4)%5]), axis=0)
            
            # train
            w = cv_train(CV_EPOCH, trainset, lr, c)
            # validation
            temp_p, temp_r, temp_f1 = cv_validation(w, validationset)
            validation_p.append(temp_p)
            validation_r.append(temp_r)
            validation_f1.append(temp_f1)
        
        mv_p, mv_r, mv_f1 = np.mean(validation_p), np.mean(validation_r), np.mean(validation_f1)
        #print("lr=%f, c=%f, average_p=%f, average_r=%f, average_f1=%f" % (lr, c, mv_p, mv_r, mv_f1))
        del validation_p
        del validation_r
        del validation_f1
        if mv_f1 > best_f1:
            best_lr, best_c, best_p, best_r, best_f1 = lr, c, mv_p, mv_r, mv_f1
print("best_lr=%f, best_c=%f, average_p=%f, average_r=%f, average_f1=%f" % (best_lr, best_c, best_p, best_r, best_f1))
# best_lr=1.000000, best_c=10.000000, average_p=0.448880, average_r=0.279076, average_f1=0.236991
del dataset
"""

#lr, c = best_lr, best_c
lr, c = 0.1, 1
trainset = pd.read_csv(DATA_DIR + 'debug.csv', header=0).to_numpy()
#testset = pd.read_csv(DATA_DIR + 'test.csv', header=0).to_numpy()

w = w0
for epoch in range(100):
    lre = lr/(epoch+1)
    np.random.shuffle(trainset)
    for datarow in trainset:
        y, x = datarow[0], np.append(datarow[1:], 1)
        if y * np.dot(w, x) <= 1:
            w = (1-lre)*w + lre*c*y*x
        else:
            w = (1-lre)*w

p, r, f1 = cv_validation(w, trainset)
print("p=%f r=%f f1=%f" % (p, r, f1))
test_mistake = 0
for test_datarow in trainset:
    y, x = test_datarow[0], np.append(test_datarow[1:], 1)
    if y * np.dot(w, x) < 0:
        test_mistake += 1

print("%d mistakes" % (test_mistake))

"""
    for test_datarow in testset:
        y, x = test_datarow[0], np.append(test_datarow[1:], 1)
        if y * np.dot(w, x) < 0:
            test_mistake += 1
    
    train_acc.append(float(train_len-train_mistake) / train_len)
    test_acc.append(float(test_len-test_mistake) / test_len)

for i in range(epoch_num):
    print("Epoch %d, training accuracy=%.2f, test accuracy=%.2f" % (i, train_acc[i], test_acc[i]))
"""





