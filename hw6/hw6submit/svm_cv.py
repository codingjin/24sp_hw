import numpy as np
import pandas as pd
import random

random.seed(43)
DATA_DIR = 'data/'
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 221)
CV_EPOCH, LARGE_NUMBER, MAX_EPOCH_NUM, THRESHOLD = 10, 1e7, 10000, 0.01

def calculate_delta(current_val, last_val):
    return abs(float(current_val-last_val)) / float(last_val)

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

def train(w, trainset, lr, C):
    cur_obj_val, last_obj_val, record_epochs, record_loss = LARGE_NUMBER, LARGE_NUMBER, [], []

    for epoch in range(MAX_EPOCH_NUM):
        lre = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1)
            if y * np.dot(w, x) <= 1:
                w = (1-lre)*w + lre*C*y*x
            else:
                w = (1-lre)*w
        
        loss = 0.0
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1)
            if y * np.dot(w, x) <= 1:
                loss += 1 - y * np.dot(w, x)
        
        record_epochs.append(epoch+1)
        record_loss.append(loss)

        last_obj_val = cur_obj_val
        cur_obj_val = 0.5*np.dot(w, w) + C*loss
        delta = calculate_delta(cur_obj_val, last_obj_val)
        if delta < THRESHOLD:
            print("epoch=%d cur_obj_val=%d delta=%.4f Done!" % (epoch+1, cur_obj_val, delta))
            return w, record_epochs, record_loss

    print("epoch=%d cur_obj_val=%d delta=%.4f Done!" % (MAX_EPOCH_NUM, cur_obj_val, delta))
    return w, record_epochs, record_loss

def test(w, testset):
    tp, tn, fp, fn = 0, 0, 0, 0
    for datarow in testset:
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

#"""
# do CV to attain the optimal parameters
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

