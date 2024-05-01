import numpy as np
import pandas as pd
import random

random.seed(43)
np.random.seed(43)
DATA_DIR = 'd5data/'
w0 = np.random.uniform(-0.01, 0.01, 101)
CV_EPOCH, LARGE_NUMBER, MAX_EPOCH_NUM, THRESHOLD = 10, 1e7, 10000, 0.01

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
    tp, fp, fn = 0, 0, 0
    for datarow in validationset:
        y, x = datarow[0], np.append(datarow[1:], 1)
        if y * np.dot(w, x) < 0:
            if y==1:    fn += 1
            else:       fp += 1
        else:
            if y==1:    tp += 1

    if tp+fp == 0:  p=1
    else:           p = float(tp)/float(tp+fp)
    r = float(tp)/float(tp+fn)
    if p+r==0:  f1 =0
    else:   f1 = 2*p*r/(p+r)
    return (p, r, f1)

# do CV to attain the optimal parameters
dataset = []
dataset.append(pd.read_csv(DATA_DIR + 'CVfolder/fold0.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolder/fold1.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolder/fold2.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolder/fold3.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolder/fold4.csv', header=0).to_numpy())

LR_LIST, C_LIST = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001], [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
best_lr, best_c, best_p, best_r, best_f1 = 1.0, 100.0, 0.0, 0.0, -0.01
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
        del validation_p
        del validation_r
        del validation_f1
        if mv_f1 > best_f1:
            best_lr, best_c, best_p, best_r, best_f1 = lr, c, mv_p, mv_r, mv_f1
print("best_lr=%f, best_c=%f, average_p=%f, average_r=%f, average_f1=%f" % (best_lr, best_c, best_p, best_r, best_f1))

