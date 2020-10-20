import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_data = pd.read_csv('Detail_Cars.csv')

df_data = df_data.replace('?',np.nan)

df_data.select_dtypes(include=['float','int'])

df_data['price'] = pd.to_numeric(df_data['price'], errors='coerce')
df_data['horsepower'] = pd.to_numeric(df_data['horsepower'], errors='coerce')
df_data['bore'] = pd.to_numeric(df_data['bore'], errors='coerce')
df_data['stroke'] = pd.to_numeric(df_data['stroke'], errors='coerce')
df_data['peak-rpm'] = pd.to_numeric(df_data['peak-rpm'], errors='coerce')

cylin_dict = {'two': 2, 'three': 3, 'four':4, 'five':5 , 'six':6, 'eight':8, 'twelve':12}

df_data['num-of-cylinders'].replace(cylin_dict, inplace=True)

df_data = df_data.drop('normalized-losses', axis=1)

df_data = pd.get_dummies(df_data)

df_data = df_data.dropna()


X = df_data.drop('price', axis=1)

y = df_data['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha = 0.5, normalize=True)
model_lasso.fit(X_train, y_train)

model_lasso.score(X_train, y_train)

y_predict = model_lasso.predict(X_test)

r_square_lasso = model_lasso.score(X_test, y_test)
print(r_square_lasso)

#Graph
plt.plot(y_predict, label='predict')
plt.plot(y_test.values, label='actual')
plt.ylabel('price')
plt.legend()
plt.show()


feature_col = X_train.columns
coef_val = pd.Series(model_lasso.coef_, feature_col).sort_values()
print(coef_val)

#Bar Coef Plot
coef_val.plot(kind='bar', title='coefficients plot Lasso')
















