import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.compose import TransformedTargetRegressor
import os
datafile = pd.read_csv(os.path.join(os.getcwd(), "train.csv"), keep_default_na=False)

datafile.drop('Id', inplace=True, axis=1)

y1 = datafile['SalePrice'].to_numpy()
datafile.drop('SalePrice', inplace=True, axis=1)

X1 = datafile.to_numpy()

to = time()


enc = OrdinalEncoder()
X1 = enc.fit_transform(X1)

X1 = X1.astype(np.float64)
y1 = y1.astype(np.float64)

std = StandardScaler()
X1 = std.fit_transform(X1, y1)

# hidden_layer_sizes = καθορίζει το πόσα hidden layers θέλω να έχω και σε τι βάθος
net = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=200)
reg = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())

score = cross_val_score(reg, X1, y1, cv=10)
print('Mean score of R^2 is: ' + str(score.mean()))
print('\n')
print('R^2 Score is: ' + str(score))
print('\n')


print('\n')
print(time()-to)
