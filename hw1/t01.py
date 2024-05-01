from data import Data
import numpy as np
import math
DATA_DIR = 'data/'


data_obj = Data(fpath=DATA_DIR+'t.csv')
#print("The entropy of the data is", get_entropy(data_obj))

print(data_obj.get_attribute_possible_vals(attribute_name='stalk-root'))

data_obj.set_missing_feature(attribute_name='stalk-root', to_set_value='b')

possibles = data_obj.get_attribute_possible_vals(attribute_name='stalk-root')
for p in possibles:
    print(p)




