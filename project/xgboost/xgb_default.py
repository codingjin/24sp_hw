import numpy as np
from scipy.stats import uniform, randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import random
import pandas as pd
#from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

random.seed(43)
np.random.seed(43)
DATA_DIR = '../data/tfidf_misc/'
train_file = DATA_DIR + 'tfidf.misc.train.csv'
eval_file = DATA_DIR + 'tfidf.misc.eval.csv'

df_train = pd.read_csv(train_file)
trainset_x = df_train.drop('label', axis=1)
trainset_y = df_train['label']

xgb_model = XGBClassifier(objective='binary:logistic')
xgb_model.fit(trainset_x, trainset_y)

df_eval = pd.read_csv(eval_file)
evalset_x = df_eval.drop('label', axis=1)
pred_evalset_y = xgb_model.predict(evalset_x)
print("example_id,label")
index = 0
for py in pred_evalset_y:
    print("%d,%d" % (index, py))
    index += 1
