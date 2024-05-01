import numpy as np
import pandas as pd
import os
import csv

SOURCE_DATA_DIR, NEW_DATA_DIR = "data/misc/", "data/misc/encoded/"
TRAIN_SIZE, TEST_SIZE, EVAL_SIZE = 17500, 2250, 5250
TRAIN_BOUND, TEST_BOUND = TRAIN_SIZE, TRAIN_SIZE+TEST_SIZE
trainfile, testfile, evalfile = SOURCE_DATA_DIR + "misc.train.csv", SOURCE_DATA_DIR + "misc.test.csv", SOURCE_DATA_DIR + "misc.eval.csv"

# Write wholeset
train_df = pd.read_csv(trainfile)
wholelist = [list(train_df)]
traindata_list = train_df.values.tolist()
for row in traindata_list:
    wholelist.append(row)
del train_df
del traindata_list

test_df = pd.read_csv(testfile)
testdata_list = test_df.values.tolist()
for row in testdata_list:
    wholelist.append(row)
del test_df
del testdata_list

eval_df = pd.read_csv(evalfile)
evaldata_list = eval_df.values.tolist()
for row in evaldata_list:
    wholelist.append(row)
del eval_df
del evaldata_list

os.mkdir(NEW_DATA_DIR)
wholeset_file = NEW_DATA_DIR + "wholeset.csv"
with open(wholeset_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in wholelist:
        writer.writerow(row)
del wholelist

# perform one_hot_encode on the wholeset
wholedata = pd.read_csv(wholeset_file)
wholedata_encoded = pd.get_dummies(wholedata, columns=['defendant_age', 'defendant_gender', 'num_victims', 'victim_genders', 'offence_category', 'offence_subcategory'], dtype=float)

# get header
header = list(wholedata_encoded)
wholedata_list = wholedata_encoded.values.tolist()

# encoded trainfile
encoded_trainfile = NEW_DATA_DIR + "misc.encoded.train.csv"
with open(encoded_trainfile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in wholedata_list[:TRAIN_BOUND]:
        writer.writerow(row)

# encoded testfile
encoded_testfile = NEW_DATA_DIR + "misc.encoded.test.csv"
with open(encoded_testfile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in wholedata_list[TRAIN_BOUND:TEST_BOUND]:
        writer.writerow(row)

# encoded evalfile
encoded_evalfile = NEW_DATA_DIR + "misc.encoded.eval.csv"
with open(encoded_evalfile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in wholedata_list[TEST_BOUND:]:
        writer.writerow(row)

