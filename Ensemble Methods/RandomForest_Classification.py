#Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('BankNote_Authentication.csv')

X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
rf_c = RandomForestClassifier(n_estimators = 200, random_state = 2)
rf_c.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred_test = rf_c.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(test_acc)


from matplotlib.colors import ListedColormap
import numpy as np
#Define Variables
clf = rf_c
h = 0.01
X_plot, z_plot = X_test, y_test

#Standard Template to draw graph
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
             alpha = 0.7, cmap = ListedColormap(('red', 'green')))


for i, j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
                c = ['red', 'green'][i], cmap = ListedColormap(('red', 'green')), label = j)
   #X[:, 0], X[:, 1] 
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Random Forest Classification')
plt.xlabel('variance')
plt.ylabel('skewness')
plt.legend()

plt.show()