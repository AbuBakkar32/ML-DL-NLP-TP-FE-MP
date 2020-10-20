# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Job_Exp.csv')
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.ensemble import GradientBoostingRegressor

gb_r = GradientBoostingRegressor(n_estimators = 500, max_depth = 6, learning_rate = 0.01)
gb_r.fit(X_train, y_train)

print(gb_r.score(X_train, y_train))

print(gb_r.score(X_test, y_test))

#Graph 
X_dt = np.arange(min(X), max(X), 0.1)
X_dt = X_dt.reshape(len(X_dt), 1)
plt.scatter(X, y, color = 'black')
plt.plot(X_dt, gb_r.predict(X_dt), color = 'red' )
plt.ylabel('Getting Job chance (%)')
plt.xlabel('years of Exp')
plt.show()

