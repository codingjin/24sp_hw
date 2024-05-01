import numpy as np
import pandas as pd
import random

random.seed(43)
DATA_DIR = 'd10data/'
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 101)
LARGE_NUMBER, MAX_EPOCH_NUM, THRESHOLD = 1e7, 1000, 0.001

def calculate_delta(current_val, last_val):
    return abs(float(current_val-last_val)) / float(last_val)

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
            print("epoch=%d cur_obj_val=%d delta=%.5f Done!" % (epoch+1, cur_obj_val, delta))
            return w, record_epochs, record_loss

    print("epoch=%d cur_obj_val=%d delta=%.4f Done!" % (MAX_EPOCH_NUM, cur_obj_val, delta))
    return w, record_epochs, record_loss

def test(w, testset):
    tp, fp, fn = 0, 0, 0
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

# best_lr=0.000010, best_c=1000.000000, average_p=0.684211, average_r=0.370518, average_f1=0.480464
lr, c = 0.00001, 1000.0
trainset = pd.read_csv(DATA_DIR + 'tree10_train.csv', header=0).to_numpy()
testset = pd.read_csv(DATA_DIR + 'tree10_test.csv', header=0).to_numpy()
w, record_epochs, record_loss = train(w0, trainset, lr, c)
p, r, f1 = test(w, testset)
print("On Test dataset: P=%f R=%f and F1=%f" % (p, r, f1))

# Plot the Epoch & Loss
import matplotlib.pyplot as plt
plt.figure(figsize=(50, 50))
plt.plot(record_epochs, record_loss, marker='o')
plt.xticks(record_epochs)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
