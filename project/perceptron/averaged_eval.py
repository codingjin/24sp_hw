import numpy as np
import pandas as pd
import random

DATA_DIR, TRAIN_EPOCH = '../data/tfidf_misc/', 9
random.seed(43)
np.random.seed(43)
w0, b0 = np.random.uniform(-0.01, 0.01, 10267), np.random.uniform(-0.01, 0.01)

def train(epoch_num, trainset, lr):
    w, b, aw, ab = w0, b0, 0.0, 0.0
    for epoch in range(epoch_num):
        lrp = lr/(epoch+1) 
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            if y == 0:
                y = -1
            
            if y*(np.dot(x, w)+b) < 0:
                w += lrp*y*x
                b += lrp*y
            
            aw, ab = aw+w, ab+b
    return (aw, ab)


lr = 100.0 # optimal
trainset = pd.read_csv(DATA_DIR + 'tfidf.misc.train.csv', header=0).to_numpy()
aw, ab = train(TRAIN_EPOCH, trainset, lr)

print("example_id,label")
df_eval_data = pd.read_csv(DATA_DIR + 'tfidf.misc.eval.csv')
for index, row in df_eval_data.iterrows():
    label, x = 1, row[1:]
    if np.dot(x, aw) + ab < 0:
        label = 0
    print("%d,%d" % (index, label))
