from data import Data
import numpy as np
import math
import pandas as pd
#DATA_DIR = 'debugdata/'
DATA_DIR = 'data/'
global common_stalkroot_value

class Node:

    def __init__(self, attributes=None, feature=None, label=None, ig=0):
        self.attributes = attributes
        self.feature = feature
        self.label = label
        self.ig = ig
        self.children = {}
        self.children_num = 0

    def set_attributes(self, attributes):
        if attributes == None:
            print("Invalid attributes setting")
            exit()
        self.attributes = attributes

    def get_attributes(self):
        return self.attributes

    def set_ig(self, ig):
        if ig == None:
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
        self.children[feature_value] = Node
        self.children_num += 1

    def get_child(self, feature_value):
        return self.children[feature_value]

    def get_children(self):
        return self.children


def get_entropy(data_obj):
    if data_obj==None or data_obj.__len__()==0:
        print("get_entropy() error due to None or Empty data_obj as input")
        exit()
    data_size = data_obj.__len__()
    p0 = float(data_obj.get_row_subset(attribute_name='label', attribute_value='p').__len__()) / data_size
    p1 = float(data_obj.get_row_subset(attribute_name='label', attribute_value='e').__len__()) / data_size
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
        p0 = float(sub_data_obj.get_row_subset(attribute_name='label', attribute_value='p').__len__()) / sub_data_size
        p1 = float(sub_data_obj.get_row_subset(attribute_name='label', attribute_value='e').__len__()) / sub_data_size
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

def ID3(data_obj, attributes):
    if data_obj==None or data_obj.__len__()==0:
        print("ID3 error due to None or Empty input!")
        exit()
    
    if len(np.unique(data_obj.get_column('label'))) == 1:
        return Node(label=data_obj.get_column('label')[0])
    
    labels_ = data_obj.get_column('label')
    labels, counts = np.unique(labels_, return_counts=True)
    common_label = labels[np.argmax(counts)]
    
    if attributes==None or len(attributes)==0:
        return Node(label=common_label)

    maxig = [-1]
    best_attribute = get_best_attribute(data_obj, attributes, maxig)
    #print("best_attribute=", best_attribute, " maxig=", maxig[0])
    if maxig[0] < 0:
        print("get_best_attribute()exception for getting invalid maxig ", maxig[0])
        exit()
    
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
            subnode = Node(label=common_label)
            node.add_child(subnode, fv)
        else:
            subnode = ID3(sub_data_obj, sub_attributes)
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
    return walk(datarow, node.get_child(feature_value=feature_value))


attributes = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'gill-attachment', 'gill-spacing', 'gill-size', 
              'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
              'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat',]

attributes_dict = {
    'cap-shape' : ['b', 'c', 'x', 'f', 'k', 's'], #bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
    'cap-surface' : ['f', 'g', 'y', 's'], #fibrous=f,grooves=g,scaly=y,smooth=s
    'cap-color' : ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'], #brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
    'bruises' : ['t', 'f'], #bruises=t,no=f 
    'gill-attachment' : ['a', 'd', 'f', 'n'], #attached=a,descending=d,free=f,notched=n 
    'gill-spacing' : ['c', 'w', 'd'], #close=c,crowded=w,distant=d 
    'gill-size' : ['b', 'n'], #broad=b,narrow=n 
    'gill-color' : ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'], #black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y  
    'stalk-shape' : ['e', 't'], #enlarging=e,tapering=t 
    'stalk-root' : ['b', 'c', 'u', 'e', 'z', 'r'], #bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?  
    'stalk-surface-above-ring' : ['f', 'y', 'k', 's'], #fibrous=f,scaly=y,silky=k,smooth=s 
    'stalk-surface-below-ring' : ['f', 'y', 'k', 's'], #fibrous=f,scaly=y,silky=k,smooth=s 
    'stalk-color-above-ring' : ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], #brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y  
    'stalk-color-below-ring' : ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], #brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
    'veil-type' : ['p', 'u'], #partial=p,universal=u
    'veil-color' : ['n', 'o', 'w', 'y'], #brown=n,orange=o,white=w,yellow=y 
    'ring-number' : ['n', 'o', 't'], #none=n,one=o,two=t
    'ring-type' : ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'], #cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
    'spore-print-color' : ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], #black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
    'population' : ['a', 'c', 'n', 's', 'v', 'y'], #abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
    'habitat' : ['g', 'l', 'm', 'p', 'u', 'w', 'd'], #grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
}

data_obj = Data(fpath=DATA_DIR+'train.csv')

stalkroot_values = data_obj.get_column(attribute_names='stalk-root')
vs, counts = np.unique(stalkroot_values, return_counts=True)
common_stalkroot_value = vs[np.argmax(counts)]
data_obj.set_missing_feature(attribute_name='stalk-root', to_set_value=common_stalkroot_value)
#print("The most common stalk-root feature value in the training data is ", common_stalkroot_value)
print("The entropy of the data is", get_entropy(data_obj))
#print(data_obj.get_attribute_possible_vals(attribute_name='stalk-root'))

node = ID3(data_obj, attributes)
print("The root feature that is select is", node.get_feature())
print("Information gain for the root feature is", node.get_ig())
print("Maximum depth of the tree is", get_depth(node))

df_train_data = pd.read_csv(DATA_DIR+'train.csv')
df_train_data.loc[df_train_data['stalk-root']=='?', 'stalk-root'] = common_stalkroot_value
train_data_size = df_train_data.shape[0]
train_correct = 0
for index, row in df_train_data.iterrows():
    #print(row)
    if row['label']==walk(row, node):
        train_correct += 1
print("Accuracy on the training set is %.2f%%" % (float(train_correct)*100/train_data_size))

df_test_data = pd.read_csv(DATA_DIR+'test.csv')
df_test_data.loc[df_test_data['stalk-root']=='?', 'stalk-root'] = common_stalkroot_value
test_data_size = df_test_data.shape[0]
test_correct = 0
for index, row in df_test_data.iterrows():
    #print(row)
    if row['label']==walk(row, node):
        test_correct += 1
print("Accuracy on the test set is %.2f%%" % (float(test_correct)*100/test_data_size))

