import numpy as np
import pandas as pd
import random
import math

random.seed(43)
DATA_DIR = 'data/'
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 221)
INF, SMALL_POWER, MAX_EPOCH_NUM, THRESHOLD = 1e6, -6, 100000, 0.001

def sigmoid_func(z):
    if z>=6.0:  return 1.0
    elif z<=-6.0:   return 0
    else:   return 1.0 / (1.0+math.exp(-z))

def log_func(x):
    if x<1e-4:  return SMALL_POWER
    else:   return math.log(x)

def calculate_delta(current_val, last_val):
    return abs(float(current_val-last_val)) / float(last_val)

def train(w, trainset, lr, sigma):
    cur_obj_val, last_obj_val, record_epochs, record_loss = INF, INF, [], []
    for epoch in range(MAX_EPOCH_NUM):
        lre = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = float(datarow[0]), np.append(datarow[1:], 1.0)
            w = (1.0 - 2.0*lre/sigma)*w + y*x*lre*sigmoid_func(-y*np.dot(w,x))
        
        loss = 0.0
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1.0)
            loss += -log_func(sigmoid_func(y*np.dot(w,x)))

        record_epochs.append(epoch+1)
        record_loss.append(loss)
        last_obj_val = cur_obj_val
        cur_obj_val = loss + 1.0*np.dot(w,w)/sigma
        delta = calculate_delta(cur_obj_val, last_obj_val)
        if delta < THRESHOLD:
            print("epoch=%d cur_obj_val=%d delta=%f Done!" % (epoch+1, cur_obj_val, delta))
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
    if tp+fp == 0:  p=1
    else:           p = float(tp)/float(tp+fp)
    r = float(tp)/float(tp+fn)
    if p+r==0:  f1 =0
    else:   f1 = 2*p*r/(p+r)

    return (p, r, f1)

# best_lr=0.100000, best_sigma=10000.000000, average_p=0.659898, average_r=0.353664, average_f1=0.459713
lr, sigma = 0.1, 10000.0
trainset = pd.read_csv(DATA_DIR + 'train.csv', header=0).to_numpy()
testset = pd.read_csv(DATA_DIR + 'test.csv', header=0).to_numpy()

w, record_epochs, record_loss = train(w0, trainset, lr, sigma)
p, r, f1 = test(w, testset)
print("On Test dataset: P=%f R=%f and F1=%f" % (p, r, f1))

# Plot the Epoch & Loss
import matplotlib.pyplot as plt
plt.figure(figsize=(250, 250))
plt.plot(record_epochs, record_loss, marker='o')
plt.xticks(record_epochs)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
