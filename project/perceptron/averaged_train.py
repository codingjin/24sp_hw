import numpy as np
import pandas as pd
import random

DATA_DIR, TRAIN_EPOCH = '../data/tfidf_misc/', 200
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

def test(w, b, validationset):
    validation_size, mistake = len(validationset), 0
    for datarow in validationset:
        y, x = datarow[0], datarow[1:]
        if y == 0:
            y = -1
        
        if y * (np.dot(x, w) + b) < 0:
            mistake += 1
    return float(validation_size-mistake) / validation_size

def advanced_train(epoch_num, trainset, testset, lr):
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
        
        # Training in this epoch is done, now check accuracy
        train_accuracy = test(aw, ab, trainset)
        test_accuracy = test(aw, ab, testset)
        print("Epoch=%d train_accuracy=%.4f test_accuracy=%.4f" % (epoch+1, train_accuracy, test_accuracy))
    #return (best_aw, best_ab)

lr = 100.0 # optimal

#lr = 1000.0 # Overfitting! Best: Epoch=8 train_accuracy=0.8946 test_accuracy=0.8502
#lr = 500.0 # Overfitting! Best: Epoch=8 train_accuracy=0.8947 test_accuracy=0.8502
#lr = 300.0 # Overfitting! Best: Epoch=8 train_accuracy=0.8947 test_accuracy=0.8502
#lr = 200.0 # Overfitting! Best: Epoch=8 train_accuracy=0.8947 test_accuracy=0.8516
#lr = 100.0 # epoch=100? Overfitting! Best: Epoch=9 train_accuracy=0.8975 test_accuracy=0.8520
#lr = 50.0  # epoch=100? Overfitting! Best: Epoch=11 train_accuracy=0.9041 test_accuracy=0.8484
#lr = 10.0 # Overfitting! Best: Epoch=4 train_accuracy=0.8751 test_accuracy=0.8480
#lr = 1.0 # epoch=100? Overfitting! Best: Epoch=5 train_accuracy=0.8819 test_accuracy=0.8480
#lr = 0.1 # => epoch=100, Epoch=3 train_accuracy=0.8643 test_accuracy=0.8458
#lr = 0.01 # Epoch=7 train_accuracy=0.8885 test_accuracy=0.8502
#lr = 0.001 # => epoch=200, test_accuracy=0.78
#lr = 0.0001 # => epoch=10, test_accuracy=0.7142
trainset, testset = pd.read_csv(DATA_DIR + 'tfidf.misc.train.csv', header=0).to_numpy(), pd.read_csv(DATA_DIR + 'tfidf.misc.test.csv', header=0).to_numpy()
advanced_train(TRAIN_EPOCH, trainset, testset, lr)


