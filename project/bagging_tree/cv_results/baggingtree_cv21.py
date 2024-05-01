from data import Data
import numpy as np
import math
import pandas as pd
import random
import csv
import os
from collections import Counter

TREE_NUM = 21
random.seed(43)
np.random.seed(43)
DATA_DIR, LIMIT_DEPTH = '../data/misc/', 1
traindata_obj, testdata_obj, evaldata_obj = Data(fpath = DATA_DIR + 'misc.train.csv'), Data(fpath = DATA_DIR + 'misc.test.csv'), Data(fpath = DATA_DIR + 'misc.eval.csv')
attributes = ['defendant_age', 'defendant_gender', 'num_victims', 'victim_genders', 'offence_category', 'offence_subcategory']
attributes_dict = {}
for attr in attributes:
    attributes_dict[attr] = list(
                                            set(list(traindata_obj.get_attribute_possible_vals(attribute_name=attr)) 
                                                + list(testdata_obj.get_attribute_possible_vals(attribute_name=attr))
                                                + list(evaldata_obj.get_attribute_possible_vals(attribute_name=attr))
                                                )
                                )
del testdata_obj
del evaldata_obj
del traindata_obj

def get_common(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

class Node:
    def __init__(self, attributes=None, feature=None, label=None, ig=0, level=0):
        self.attributes = attributes
        self.feature = feature
        self.label = label
        self.ig = ig
        self.children = {}
        self.children_num = 0
        self.level = level

    def set_attributes(self, attributes):
        if attributes == None:
            print("Invalid attributes setting")
            exit()
        self.attributes = attributes

    def get_attributes(self):
        return self.attributes

    def set_ig(self, ig):
        if ig==None or ig<0:
            print("Invalid ig setting")
            exit()
        self.ig = ig
    
    def get_ig(self):
        return self.ig
    
    def set_label(self, label=None):
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_feature(self, feature=None):
        self.feature = feature
    
    def get_feature(self):
        return self.feature
    
    def get_children_num(self):
        return self.get_children_num
    
    def add_child(self, Node, feature_value):
        if Node == None:
            print("Exception: adding None child")
            exit()
        self.children[str(feature_value)] = Node
        self.children_num += 1

    def get_child(self, feature_value):
        if (self.children_num != 0):    return self.children[str(feature_value)]
        else:                           return None

    def get_children(self):
        return self.children

    def get_level(self):
        return self.level


def get_entropy(data_obj):
    if data_obj==None or data_obj.__len__()==0:
        print("get_entropy() error due to None or Empty data_obj as input")
        exit()
    data_size = data_obj.__len__()
    p0 = float(data_obj.get_row_subset(attribute_name='label', attribute_value='0').__len__()) / data_size
    p1 = float(data_obj.get_row_subset(attribute_name='label', attribute_value='1').__len__()) / data_size
    if p0==0 or p1==0:
        return 0
    return float(-1)*(p0*math.log2(p0) + p1*math.log2(p1))

def get_attribute_entropy(data_obj, attribute):
    if data_obj==None or data_obj.__len__()==0 or attribute==None:
        print("get_attribute_entropy() error due to None or Empty data_obj as input")
        exit()
    data_size = data_obj.__len__()
    attribute_vals = data_obj.get_attribute_possible_vals(attribute_name=attribute)
    sub_entropy = 0.0
    for attribute_val in attribute_vals:
        sub_data_obj = data_obj.get_row_subset(attribute_name=attribute, attribute_value=attribute_val)
        sub_data_size = sub_data_obj.__len__()
        if sub_data_size==0:
            continue
        p0 = float(sub_data_obj.get_row_subset(attribute_name='label', attribute_value='0').__len__()) / sub_data_size
        p1 = float(sub_data_obj.get_row_subset(attribute_name='label', attribute_value='1').__len__()) / sub_data_size
        if p0!=0 and p1!=0:
            sub_entropy = sub_entropy + float(-1*sub_data_size)*(p0*math.log2(p0) + p1*math.log2(p1))

    return float(sub_entropy) / data_size

def get_best_attribute(data_obj, attributes, max_info_gain):
    if data_obj==None or data_obj.__len__()==0 or attributes==None or len(attributes)==0:
        print("get_best_attribute() error due to None or Empty data_obj / attributes as input")
        exit()
    s_entropy = get_entropy(data_obj)
    best_attribute = None
    maxig = -1
    for attribute in attributes:
        tmp_ig = s_entropy - get_attribute_entropy(data_obj, attribute)
        if tmp_ig > maxig:
            maxig = tmp_ig
            best_attribute = attribute

    max_info_gain[0] = maxig
    return best_attribute

def ID3(data_obj, attributes, current_level):
    if data_obj==None or data_obj.__len__()==0:
        print("ID3 error due to None or Empty input!")
        exit()
    
    if current_level > LIMIT_DEPTH:
        print("ID3() error due to depth limitation")
        exit()

    if len(np.unique(data_obj.get_column('label'))) == 1:
        return Node(label=data_obj.get_column('label')[0])
    
    labels_ = data_obj.get_column('label')
    labels, counts = np.unique(labels_, return_counts=True)
    common_label = labels[np.argmax(counts)]
    
    if current_level == LIMIT_DEPTH:
        return Node(label=common_label, level=current_level)

    if attributes==None or len(attributes)==0:
        return Node(label=common_label)

    maxig = [-1]
    best_attribute = get_best_attribute(data_obj, attributes, maxig)
    if maxig[0] < 0:   maxig[0] = 0
    
    sub_attributes = []
    for attribute in attributes:
        if attribute==best_attribute:
            continue
        sub_attributes.append(attribute)

    node = Node(feature=best_attribute, ig=maxig[0])
    feature_values = attributes_dict[best_attribute]
    for fv in feature_values:
        sub_data_obj = data_obj.get_row_subset(attribute_name=best_attribute, attribute_value=fv)
        subset_size = sub_data_obj.__len__()
        if subset_size == 0:
            subnode = Node(label=common_label, level=current_level+1)
            node.add_child(subnode, fv)
        else:
            subnode = ID3(sub_data_obj, sub_attributes, current_level=current_level+1)
            node.add_child(subnode, fv)

    return node

def get_depth(node):
    if node == None:
        print("get_depth get None Node input")
        exit()

    if node.label != None:
        return 0
    else:
        max_depth = 0
        children = node.get_children()
        for subnode in children.values():
            tmp_depth = get_depth(subnode) + 1
            if tmp_depth > max_depth:
                max_depth = tmp_depth
        return max_depth

def walk(datarow, node):
    if node==None or np.size(datarow)==0:
        print("walk exception due to empty inputs")
        exit()
    
    if node.label != None:
        return node.label

    feature = node.get_feature()
    feature_value =  datarow[feature]
    if node.label==None and node.get_child(feature_value=feature_value)==None:
        print("exception node!")
        print("feature=", feature, "feature_value=", feature_value)
    return walk(datarow, node.get_child(feature_value=feature_value))


train_df = pd.read_csv(DATA_DIR + 'misc.train.csv')
trainlist = [list(train_df)] # Header
valuelist = train_df.values.tolist() # All datarows
for row in valuelist:
    trainlist.append(row)

TRAIN_SIZE, SAMPLE_SIZE = len(trainlist)-1, int((len(trainlist)-1)*0.2)
# Step 1: Generate TREE_NUM trees each with depth=1 and sampling 20% of the training dateset
nodes = []
for i in range(TREE_NUM):
    indices = np.random.choice(TRAIN_SIZE, SAMPLE_SIZE)
    current_trainlist = [trainlist[0]]
    for index in indices:
        current_trainlist.append(trainlist[index+1])
    traindata_obj = Data(data=current_trainlist)
    current_node = ID3(traindata_obj, attributes, current_level=0)
    nodes.append(current_node)
    del traindata_obj
    del current_node
    del current_trainlist
del trainlist

df_train_data = pd.read_csv(DATA_DIR + 'misc.train.csv')
train_data_size, num_mistake = df_train_data.shape[0], 0
for index, row in df_train_data.iterrows():
    temp_results = []
    for node in nodes:  temp_results.append(int(walk(row, node)))
    common_label = get_common(temp_results)
    if int(row['label'])!=int(common_label):    num_mistake += 1
print("TREE_NUM=%d Accuracy on the training set is %.4f%%" % (TREE_NUM, 100*float(train_data_size-num_mistake)/train_data_size))


df_test_data = pd.read_csv(DATA_DIR + 'misc.test.csv')
test_data_size, num_mistake = df_test_data.shape[0], 0
for index, row in df_test_data.iterrows():
    temp_results = []
    for node in nodes:  temp_results.append(int(walk(row, node)))
    common_label = get_common(temp_results)
    if int(row['label'])!=int(common_label):    num_mistake += 1
print("TREE_NUM=%d Accuracy on the test set is %.4f%%" % (TREE_NUM, 100*float(test_data_size-num_mistake)/test_data_size))

