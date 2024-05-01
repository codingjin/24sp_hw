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
test_file = DATA_DIR + 'tfidf.misc.test.csv'

df_train = pd.read_csv(train_file)
#print(df_train.head())\
trainset_x = df_train.drop('label', axis=1)
trainset_y = df_train['label']

#print(trainset_x.head())
#print(trainset_y.head())


params = {
            'max_depth': [1, 2, 4, 8, 10, 16, 20],
            'learning_rate': [0.1, 0.2, 0.3],
        }

"""
params = {
    "learning_rate": [0.1, 0.15, 0.20, 0.25, 0.30], # default 0.1 
    "max_depth": randint(1, 30), # default 3
    "n_estimators": randint(100, 200), # default 100
    "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
"""
#xgb_model = XGBClassifier(**params)


xgb_model = XGBClassifier(objective='binary:logistic')

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=43, n_iter=60, cv=5, verbose=1, n_jobs=-1, return_train_score=True)
search.fit(trainset_x, trainset_y)
report_best_scores(search.cv_results_, 1)

"""
xgb_model.fit(trainset_x, trainset_y)
df_test = pd.read_csv(test_file)
testset_x = df_test.drop('label', axis=1)
testset_y = df_test['label']

print(xgb_model.score(testset_x, testset_y))
"""
#print(testset_y)
#for l in testset_y:
#    print(int(l))
