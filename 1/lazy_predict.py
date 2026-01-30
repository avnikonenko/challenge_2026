#pip install lazypredict
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

# boston = datasets.load_boston()
# X, y = shuffle(boston.data, boston.target, random_state=13)
# X = X.astype(np.float32)
f_train = 'MorganFprRDKit_train_protonated_0.csv'
f_test = 'MorganFprRDKit_blind_protonated_0.csv'
#train
data_train = pd.read_csv(f_train)
x_train = data_train.drop(['mol_id', 'act'], axis='columns')
y_train = data_train.loc[:,'act']
#test
data_test = pd.read_csv(f_test)
x_blind = data_test.drop(['mol_id','act'], axis='columns')


# X, y = shuffle(x_train,y_train, random_state=13)
X, y = x_train, y_train

offset = int(X.shape[0] * 0.8)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
#
chosen_regressors  = ['LinearRegression', 'Ridge', 'Lasso','TransformedTargetRegressor','AdaBoostRegressor',
                    'BayesianRidge','ElasticNet','Adjusted R-Squared',
              'DecisionTreeRegressor', 'RandomForestRegressor','ExtraTreeRegressor', 'KernelRidge',
              'GradientBoostingRegressor', 'KNeighborsRegressor', 'BaggingRegressor','XGBRegressor',
              'SVR']
# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None, regressors=REGRESSORS)
# models, predictions = reg.fit(X_train, X_test, y_train.values.ravel(), y_test)
REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] in chosen_regressors))
]

reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None, regressors=REGRESSORS)

# reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None, regressors='all')
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# print(models)
