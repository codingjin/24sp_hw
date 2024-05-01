import numpy as np
import pandas as pd
import random
import math

np.seterr(over='raise')
random.seed(43)
DATA_DIR = 'data/'
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 221)
CV_EPOCH = 10

def sigmoid_func(z):
    if z>=6.0:  return 1.0
    elif z<=-6.0:   return 0
    else:   return 1.0 / (1.0+math.exp(-z))

def cv_train(epoch_num, trainset, lr, sigma):
    w = w0
    for epoch in range(epoch_num):
        lre = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1.0)
            w = (1.0 - 2.0*lre/sigma)*w + y*x*lre*sigmoid_func(-y*np.dot(w,x))
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
    if p+r==0:  f1 =0
    else:   f1 = 2*p*r/(p+r)

    return (p, r, f1)

dataset = []
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training00.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training01.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training02.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training03.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/training04.csv', header=0).to_numpy())

LR_LIST, SIGMA_LIST = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001], [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
best_lr, best_sigma, best_p, best_r, best_f1 = 1.0, 0.1, 0.0, 0.0, -0.01
for lr in LR_LIST:
    for sigma in SIGMA_LIST:
        validation_p, validation_r, validation_f1 = [], [], []
        exception_flag = False
        for i in range(5):
            #exception_flag = False
            validationset = dataset[i]
            trainset = np.concatenate((dataset[(i+1)%5], dataset[(i+2)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+3)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+4)%5]), axis=0)
            
            # train
            try:
                w = cv_train(CV_EPOCH, trainset, lr, sigma)
            except FloatingPointError as e:
                print(f'lr={lr}, sigma={sigma}, FloatingPointError: {e}')
                exception_flag = True
                break
            if exception_flag:  break

            # validation
            temp_p, temp_r, temp_f1 = cv_validation(w, validationset)
            validation_p.append(temp_p)
            validation_r.append(temp_r)
            validation_f1.append(temp_f1)
        
        if exception_flag: continue

        mv_p, mv_r, mv_f1 = np.mean(validation_p), np.mean(validation_r), np.mean(validation_f1)
        print("lr=%f, sigma=%f, average_p=%f, average_r=%f, average_f1=%f" % (lr, sigma, mv_p, mv_r, mv_f1))
        del validation_p
        del validation_r
        del validation_f1
        if mv_f1 > best_f1:
            best_lr, best_sigma, best_p, best_r, best_f1 = lr, sigma, mv_p, mv_r, mv_f1
print("best_lr=%f, best_sigma=%f, average_p=%f, average_r=%f, average_f1=%f" % (best_lr, best_sigma, best_p, best_r, best_f1))
# best_lr=0.100000, best_sigma=10000.000000, average_p=0.659898, average_r=0.353664, average_f1=0.459713
