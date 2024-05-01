import numpy as np
import pandas as pd
import random
import math
from data import Data
import csv

random.seed(43)
DATA_DIR = 'data/'
np.random.seed(43)
"""
with open(DATA_DIR + 'debug2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    all_list = list(reader)
    #print(list(reader))
"""

#print(all_list)
#header = all_list[0]
#print(header)

#dataobj = Data(data=all_list)
#print(dataobj.get_column('label'))

#trainset = pd.read_csv(DATA_DIR+'debug2.csv', dtype=int).to_numpy()
#print(trainset)

#print("len=", len(trainset))
df_train_data = pd.read_csv(DATA_DIR+'debug01.csv', dtype=int)
print(df_train_data)

