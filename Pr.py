import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.preprocessing import PolynomialFeatures
import os

datafile = pd.read_csv(os.path.join(os.getcwd(), "train.csv"), keep_default_na=False)

datafile.drop('Id', inplace=True, axis=1)

y1 = datafile['SalePrice']
datafile.drop('SalePrice', inplace=True, axis=1)

X1 = datafile.to_numpy()

to = time()

enc = OrdinalEncoder()
X1 = enc.fit_transform(X1)


poly = PolynomialFeatures(3)
X_train_poly = poly.fit_transform(X1)

reg = linear_model.LinearRegression()
score = cross_val_score(reg, X_train_poly, y1, cv=10)
print('Mean score of R^2 is: ' + str(score.mean()))
print('\n')
print('R^2 Score is: ' + str(score))
print('\n')


print('\n')
print(time()-to)
