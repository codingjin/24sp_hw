import numpy as np
import pandas as pd
import random

DATA_DIR, CV_EPOCH = '../data/tfidf_misc/', 2
random.seed(43)
np.random.seed(43)
w0, b0 = np.random.uniform(-0.01, 0.01, 10267), np.random.uniform(-0.01, 0.01)

def cv_train(epoch_num, trainset, lr):
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

def cv_validation(w, b, validationset):
    validation_size, mistake = len(validationset), 0
    for datarow in validationset:
        y, x = datarow[0], datarow[1:]
        if y == 0:
            y = -1

        if y * (np.dot(x, w) + b) < 0:
            mistake += 1
    return float(validation_size-mistake) / validation_size

dataset = []
dataset.append(pd.read_csv(DATA_DIR + 'CVfolders/fold0.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolders/fold1.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolders/fold2.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolders/fold3.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVfolders/fold4.csv', header=0).to_numpy())

#testset = pd.read_csv(DATA_DIR + "tfidf.misc.test.csv", header=0).to_numpy()

LR_LIST = [1000.0, 100.0, 50.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
best_lr, best_accuracy = 0, -0.1
for lr in LR_LIST:
    validation_accuracies = []
    #test_accuracies = []
    for i in range(5):
        trainset = np.concatenate((dataset[(i+1)%5], dataset[(i+2)%5]), axis=0)
        trainset = np.concatenate((trainset, dataset[(i+3)%5]), axis=0)
        trainset = np.concatenate((trainset, dataset[(i+4)%5]), axis=0)
        validationset = dataset[i]

        # training
        w, b = cv_train(CV_EPOCH, trainset, lr)
        # validation
        validation_accuracies.append(cv_validation(w, b, validationset))
        #test_accuracies.append(cv_validation(w, b, testset))
        del trainset
        del validationset
    
    validation_accuracy = np.mean(validation_accuracies)

    #test_accuracy = np.mean(test_accuracies)
    if validation_accuracy > best_accuracy:
        best_accuracy, best_lr = validation_accuracy, lr

    stdvariance = np.std(validation_accuracies)
    print("lr=%.6f, cross-validation accuracy=%.6f%%, and standard deviation is %.6f, testaccuracy=%.6f%%" % (lr, validation_accuracy*100, stdvariance, test_accuracy*100))

print("The best lr = %.6f, the corresponding cross-validation accuracy = %.6f%%" % (best_lr, best_accuracy*100))
