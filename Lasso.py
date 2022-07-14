import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.linear_model import LassoCV
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
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

reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, penalty="l1", l1_ratio=1))
reg = reg.fit(X1, y1)
score2 = cross_val_score(reg, X1, y1, cv=10)
print('Mean score of R^2 is: ' + str(score2.mean()))

print('\n')
print(time()-to)
