import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
import sklearn.linear_model as lm
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from time import time
from sklearn.preprocessing import StandardScaler
import os

datafile = pd.read_csv(os.path.join(os.getcwd(), "train.csv"), keep_default_na=False)

datafile.drop('Id', inplace=True, axis=1)

y1 = datafile['SalePrice']
datafile.drop('SalePrice', inplace=True, axis=1)

X1 = datafile.to_numpy()

to = time()

enc = OrdinalEncoder()
X1 = enc.fit_transform(X1)

X1 = X1.astype(np.float64)
y1 = y1.astype(np.float64)

std = StandardScaler()
X1 = std.fit_transform(X1, y1)

gpr = GaussianProcessRegressor(alpha=0.0000000001, optimizer=str, n_restarts_optimizer=0, normalize_y=True, copy_X_train=False, random_state=None).fit(X1, y1)
score2 = cross_val_score(gpr, X1, y1, cv=10)
print('Mean score of R^2 is: ' + str(score2.mean()))
print('\n')
print('R^2 Score  is: ' + str(score2))
print('\n')


print('\n')
print(time()-to)
