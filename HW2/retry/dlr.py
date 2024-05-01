import numpy as np
import pandas as pd

DATA_DIR = 'data/'
CV_EPOCH, TRAIN_EPOCH = 10, 20
np.random.seed(43)
w0 = np.random.uniform(-0.01, 0.01, 19)
b0 = np.random.uniform(-0.01, 0.01)

def cv_train(epoch_num, trainset, lr):
    w, b = w0, b0
    for epoch in range(epoch_num):
        lrp = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            if y*(np.dot(x, w)+b) <= 0:
                w += lrp*y*x
                b += lrp*y
    return (w, b)

def cv_validation(w, b, validationset):
    validation_size, mistake = len(validationset), 0
    for datarow in validationset:
        y, x = datarow[0], datarow[1:]
        if y * (np.dot(x, w) + b) <= 0:
            mistake += 1
    return float(validation_size-mistake) / validation_size

def test(w, b, testset):
    testset_size, mistake = len(testset), 0
    for datarow in testset:
        y, x = datarow[0], datarow[1:]
        if y*(np.dot(x, w) + b) <= 0:
            mistake += 1
    return float(testset_size-mistake) / testset_size

def train(epoch_num, trainset, devset, lr):
    w, b, update_num = w0, b0, 0
    best_w, best_b, best_accuracy, accuracies = 0, 0, -0.01, []
    for epoch in range(epoch_num):
        lrp = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            if y*(np.dot(x, w) + b) <= 0:
                w, b, update_num = w+lrp*y*x, b+lrp*y, update_num+1
            
        devset_accuracy = test(w, b, devset)
        accuracies.append(devset_accuracy)
        if devset_accuracy >= best_accuracy:
            best_w = w
            best_b = b
            best_accuracy = devset_accuracy

    print("The total number of updates the learning algorithm performs on the training set is %d" % (update_num))
    print("The development set accuracy is %.2f%%" % (100*best_accuracy))
    return (best_w, best_b, accuracies)

def debug(epoch_num, lr, trainset, testset):
    w, b = w0, b0
    train_acc, train_len, test_acc, test_len = [], len(trainset), [], len(testset)
    for epoch in range(epoch_num):
        np.random.shuffle(trainset)
        train_mistake, test_mistake = 0, 0
        lrp = lr/(epoch+1)
        for datarow in trainset:
            y, x = datarow[0], datarow[1:]
            if y*(np.dot(x, w) + b) <= 0:
                w, b, train_mistake = w+lrp*y*x, b+lrp*y, train_mistake+1

        for test_datarow in testset:
            y, x = test_datarow[0], test_datarow[1:]
            if y*(np.dot(x,w)+b) <= 0:
                test_mistake += 1

        train_acc.append(float(train_len-train_mistake)/train_len)
        test_acc.append(float(test_len-test_mistake)/test_len)
    
    return (train_acc, test_acc)

print("Decaying the learning rate")

dataset = []
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/train0.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/train1.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/train2.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/train3.csv', header=0).to_numpy())
dataset.append(pd.read_csv(DATA_DIR + 'CVSplits/train4.csv', header=0).to_numpy())

LR_LIST = [1.0, 0.1, 0.01]
best_lr, best_accuracy = 0, -0.1
for lr in LR_LIST:
    validation_accuracies = []
    for i in range(5):
        trainset = np.concatenate((dataset[(i+1)%5], dataset[(i+2)%5]), axis=0)
        trainset = np.concatenate((trainset, dataset[(i+3)%5]), axis=0)
        trainset = np.concatenate((trainset, dataset[(i+4)%5]), axis=0)
        validationset = dataset[i]

        # training
        w, b = cv_train(CV_EPOCH, trainset, lr)
        # validation
        validation_accuracies.append(cv_validation(w, b, validationset))
        del trainset
        del validationset
    validation_accuracy = np.mean(validation_accuracies)
    if (validation_accuracy > best_accuracy):
        best_accuracy = validation_accuracy
        best_lr = lr
    stdvariance = np.std(validation_accuracies)
    print("lr=%.2f, cross-validation accuracy=%.2f%%, and standard deviation is %.2f" %(lr, validation_accuracy*100, stdvariance))

print("The best lr = %.2f, the corresponding cross-validation accuracy = %.2f%%" % (best_lr, best_accuracy*100))
del dataset
lr = best_lr

# Training
trainset = pd.read_csv(DATA_DIR + 'diabetes.train.csv', header=0).to_numpy()
devset = pd.read_csv(DATA_DIR + 'diabetes.dev.csv', header=0).to_numpy()
best_w, best_b, accuracies = train(TRAIN_EPOCH, trainset, devset, lr)

# Test
testset = pd.read_csv(DATA_DIR + 'diabetes.test.csv', header=0).to_numpy()
print("The test set accuracy is %.2f%%" % (100*test(best_w, best_b, testset)))

"""
# Majority baseline on test set
test_labels = testset[:, 0]
test_len = len(test_labels)
count_1 = np.count_nonzero(test_labels == 1)
count_0 = test_len - count_1
if count_1 > count_0:
    testacc = float(count_1)/test_len
else:
    testacc = float(count_0)/test_len
print("Majority baseline accuracy on test set is %.2f%%" % (100*testacc))

# Majority baseline on dev set

dev_labels = devset[:, 0]
dev_len = len(dev_labels)
count_1 = np.count_nonzero(dev_labels == 1)
count_0 = dev_len - count_1
if count_1 > count_0:
    devacc = float(count_1)/dev_len
else:
    devacc = float(count_0)/dev_len
print("Majority baseline accuracy on development set is %.2f%%" % (100*devacc))
"""

#"""
import matplotlib.pyplot as plt
epochs = [i for i in range(1, 21)]
plt.figure(figsize=(50, 50))
plt.plot(epochs, accuracies, marker='o')
plt.xticks(epochs)
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Dev Accuracy')
plt.legend()
plt.show()
#"""

# Debug
"""
DEBUG_EPOCH = 50
trainset = pd.read_csv(DATA_DIR + 'diabetes.train.csv', header=0).to_numpy()
testset = pd.read_csv(DATA_DIR + 'diabetes.test.csv', header=0).to_numpy()
epochs = [i for i in range(DEBUG_EPOCH)]
train_acc, test_acc = debug(DEBUG_EPOCH, lr, trainset, testset)

import matplotlib.pyplot as plt
plt.figure(figsize=(50, 50))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, test_acc, label='Test Accuracy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""


