import numpy as np
import pandas as pd
import random

# best_lr= 1e-08 best_c= 1e7 best_cv_validation_accuracy= 0.8286857142857142
random.seed(43)
np.random.seed(43)
DATA_DIR, LARGE_NUM, TRAIN_EPOCH = '../data/tfidf_misc/', 1e30, 1000
w0 = np.random.uniform(-0.01, 0.01, 10268)

def round_train(w, lre, C, datarow):
    y, x = datarow[0], np.append(datarow[1:], 1.0)
    if y == 0:
        y = -1
    
    if y * np.dot(w, x) <= 1:
        w = (1-lre)*w + lre*C*y*x
    else:
        w = (1-lre)*w
    w = np.clip(w, -LARGE_NUM, LARGE_NUM)
    return w

def advanced_train(w, lr, C, trainset, testset):
    best_epoch, best_train_accuracy, best_test_accuracy = -1, -0.01, -0.01
    for epoch in range(TRAIN_EPOCH):
        lre = lr/(1+epoch)
        np.random.shuffle(trainset)
        # Training
        for datarow in trainset:
            w = round_train(w, lre, C, datarow)
        
        train_accuracy, test_accuracy = test(w, trainset), test(w, testset)
        if test_accuracy > best_test_accuracy:
            best_epoch, best_train_accuracy, best_test_accuracy = epoch, train_accuracy, test_accuracy
        print("Epoch=%d train_accuracy=%.4f test_accuracy=%.4f" % (epoch+1, train_accuracy, test_accuracy))
    
    print("\nbest_epoch=%d best_train_accuracy=%.4f best_test_accuracy=%.4f" % (best_epoch+1, best_train_accuracy, best_test_accuracy))


def test(w, testset):
    num_mistakes, size = 0, len(testset)
    for datarow in testset:
        y, x = datarow[0], np.append(datarow[1:], 1.0)
        if y == 0:
            y = -1

        if y * np.dot(w, x) <= 0:
            num_mistakes += 1
    return float(size-num_mistakes) / size


# best_lr= 1e-08 best_c= 1e7 best_cv_validation_accuracy= 0.8286857142857142
lr, c = 1e-08, 1e7
trainset, testset = pd.read_csv(DATA_DIR + 'tfidf.misc.train.csv', header=0).to_numpy(), pd.read_csv(DATA_DIR + 'tfidf.misc.test.csv', header=0).to_numpy()

advanced_train(w0, lr, c, trainset, testset)
# Set EPOCH=

