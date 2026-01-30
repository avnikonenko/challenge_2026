import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd


##train=pd.read_csv('NW/train.csv',sep=';',decimal=',', encoding = "cp1251")
##test=pd.read_csv('NW/test.csv',sep=';',decimal=',', encoding = "cp1251")

# f_train = '/home/sany/projects/rf/MorganFprRDKit_train_0.csv'
# f_test = '/home/sany/projects/rf/MorganFprRDKit_blind.csv'

f_train = '/mnt/CCC417ABC4179734/Ivanova/challenge_2026/1/MorganFprRDKit_chembl_stereo_classes_fixed_0.csv'
f_test = '/mnt/CCC417ABC4179734/Ivanova/challenge_2026/1/MorganFprRDKit_blind_stereo_0.csv'

data_train = pd.read_csv(f_train)
x_train = data_train.drop(['mol_id', 'act'], axis='columns')
y_train = data_train.loc[:,'act']

data_test = pd.read_csv(f_test)
x_test = data_test.drop(['mol_id','act'], axis='columns')


def nw(x_train, y_train):
    model = Sequential()
    model.add(Dense(80, input_dim=len(x_train.columns), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear'))
    ##
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    t = model.fit(x_train, y_train,
                  epochs=1000,
                  batch_size=100,
                  validation_split=0.2)

    return model

m=nw(x_train, y_train)
y= m.predict(x_test)


data_test.loc[:,'act'] = y
mol_y = data_test.loc[:,['mol_id','act']]
mol_y = mol_y.sort_values('act',ascending=False)
#
mol_pred = mol_y.iloc[0:100,0]
mol_pred.to_csv('correct_answers_NP.txt',index=False)
# mol_test = pd.read_csv('/media/sany/Workplace/projects/2023_challenge/correct_answers_NP.txt')['mol_id'].to_list()
# correct_all = len(mol_test)
#
# pred_vs_test = np.intersect1d(mol_test, mol_pred)
# res = len(pred_vs_test) / correct_all
# print(res)

# for i in range(20):
#     m = 0
#     while i < 2 and m < 2:
#         score, model = nw(x_train, y_train, x_test, y_test)
#         m += 1
#         print('__*****____', score, m, i)
#         if score[1] > n:
#             n = score[1]
#             print(n)
#             print('saving')
#             model.save('NW/my_model.h5')
#             break
# print(n)

##model.save('NW/my_model.h5')
