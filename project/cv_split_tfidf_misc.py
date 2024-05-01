import pandas as pd
from sklearn.model_selection import KFold
import os

DATA_DIR = 'data/tfidf_misc/'
CV_DIR = DATA_DIR + "CVfolders/"
os.mkdir(CV_DIR)

df = pd.read_csv(DATA_DIR + "tfidf.misc.train.csv")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=43)

folds = []
for train_index, test_index in kf.split(df):
    fold = df.iloc[test_index]
    folds.append(fold)

for i, fold in enumerate(folds):
    fold.to_csv(f'{CV_DIR}fold{i}.csv', index=False)
