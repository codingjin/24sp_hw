import numpy as np
import pandas as pd
import random

random.seed(43)
np.random.seed(43)
DATA_DIR, LARGE_NUM, CV_EPOCH = '../data/tfidf_misc/', 1e31, 10
w0 = np.random.uniform(-0.01, 0.01, 10268)

def cv_train(epoch_num, trainset, lr, C):
    w = w0
    for epoch in range(epoch_num):
        lre = lr/(epoch+1)
        np.random.shuffle(trainset)
        for datarow in trainset:
            y, x = datarow[0], np.append(datarow[1:], 1.0)
            if y == 0:
                y = -1
            
            # update
            if y * np.dot(w, x) <= 1:
                w = (1-lre)*w + lre*C*y*x
            else:
                w = (1-lre)*w
            w = np.clip(w, -LARGE_NUM, LARGE_NUM)
            
    return w

# return validation accuracy
def cv_validation(w, validationset):
    validation_size, num_mistake = len(validationset), 0
    for datarow in validationset:
        y, x = datarow[0], np.append(datarow[1:], 1.0)
        if y == 0:
            y = -1
        
        if y * np.dot(w, x) <= 0:
            num_mistake += 1
    
    return float(validation_size - num_mistake) / float(validation_size)

# do CV to attain the optimal parameters
dataset = []
for i in range(5):
    dataset.append(pd.read_csv(DATA_DIR + f'CVfolders/fold{i}.csv', header=0).to_numpy())

#LR_LIST, C_LIST = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001], [1000000000.0, 100000000.0, 10000000.0, 1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
#LR_LIST, C_LIST = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000001], [1e18, 1e17, 1e16, 1e15, 1e14, 1e13, 1e12, 1e11, 1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2]
LR_LIST, C_LIST = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], [1e8, 1e7, 1e6, 1e5, 1e4, 1e3]
best_lr, best_c, best_accuracy = 5.0, 10000.0, -0.01
for lr in LR_LIST:
    three_stop_condtion = 0
    for c in C_LIST:
        validation_accuracies, stop_condition = [], 0
        for i in range(5):
            validationset = dataset[i]
            trainset = np.concatenate((dataset[(i+1)%5], dataset[(i+2)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+3)%5]), axis=0)
            trainset = np.concatenate((trainset, dataset[(i+4)%5]), axis=0)
            # train
            w = cv_train(CV_EPOCH, trainset, lr, c)
            # validation
            temp_accuracy = cv_validation(w, validationset)
            if temp_accuracy < 0.65:
                training_accuracy = cv_validation(w, trainset)
                print("lr=", lr, "c=", c, "are bad hyperparameters! training_accuracy=", training_accuracy, "cv_accuracy=", temp_accuracy)
                #print("lr=%f, c=%f are bad hyperparameters! training_accuracy=%f cv_accuracy=%f" % (lr, c, training_accuracy, temp_accuracy))
                stop_condition = 1
                three_stop_condtion += 1
                break
            validation_accuracies.append(temp_accuracy)

        if stop_condition == 1:
            if three_stop_condtion == 3:
                break
            continue
        else:
            three_stop_condtion = 0

        mean_validation_accuracy = np.mean(validation_accuracies)
        print("lr=", lr, "c=", c, "cv_validation_accuracy=", mean_validation_accuracy)
        if mean_validation_accuracy > best_accuracy:
            best_lr, best_c, best_accuracy = lr, c, mean_validation_accuracy

print("best_lr=", best_lr, "best_c=", best_c, "best_cv_validation_accuracy=", best_accuracy)

# Assume first:
# lr=5e-5, c=1e4, average_validation_accuracy=0.8269
# But this one is better: lr=1e-7, c=1e6, average_validation_accuracy=0.8271
