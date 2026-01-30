from rdkit import Chem
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit

# f_train = '/home/sany/projects/rf/MorganFprRDKit_train_0.csv'
f_train = '/media/sany/Workplace/projects/2023_challenge/MorganFprRDKit_train_protonated_0.csv'
f_test = '/media/sany/Workplace/projects/2023_challenge/MorganFprRDKit_blind_protonated_0.csv'

data_train = pd.read_csv(f_train)
x_train = data_train.drop(['mol_id', 'act'], axis='columns')
y_train = data_train.loc[:,'act']

data_test = pd.read_csv(f_test)
x_test = data_test.drop(['mol_id','act'], axis='columns')

param = {
    #'max_features': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 'log2', 'sqrt', 'auto', None]
    'max_features': [0.1, 0.2, 0.3, 'log2', 'sqrt']
}

seed = 120
n_estimators = 250

cv = KFold(n_splits=5, shuffle=True, random_state=seed)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
model =  RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
# model =
model = GridSearchCV(model, param, cv=cv, n_jobs=3).fit(X=x_train, y=y_train)  # model will be rebuilt on a whole data set

# with open('rf_class', 'wb') as f:
#     joblib.dump(model, f, protocol=-1)

res = model.predict(x_test)

data_test.loc[:,'act'] = res
data_test.loc[:,['act','mol_id']].to_csv('/media/sany/Workplace/projects/2023_challenge/blind_res_class_250')
