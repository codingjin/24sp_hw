from data import Data
import numpy as np
DATA_DIR = 'data/'

data = np.loadtxt(DATA_DIR + 'train.csv', delimiter=',', dtype=str)
data_obj = Data(data = data)
labels = data_obj.get_column(attribute_names='label')
values, counts = np.unique(labels, return_counts=True)
print("The most common label in the training data is ", values[np.argmax(counts)])
common_label = values[np.argmax(counts)]
data_size = data_obj.__len__()
print("The training accuracy is %.2f%%" % (float(100*counts[np.argmax(counts)])/data_size))


data = np.loadtxt(DATA_DIR + 'test.csv', delimiter=',', dtype=str)
data_obj = Data(data = data)
data_size = data_obj.__len__()
correct_num = data_obj.get_row_subset(attribute_name='label', attribute_value=common_label).__len__()
print("The test accuracy is %.2f%%" % (float(100*correct_num)/data_size))

