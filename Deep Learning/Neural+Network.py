import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data

#Read the data
df = pd.read_csv("pima-data.csv")

#Check the Correlation
#df.corr()
#Delete the correlated feature
del df['skin']

#Data Molding
diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

#Splitting the data
from sklearn.model_selection import train_test_split

#This will copy all columns from 0 to 7(8 - second place counts from 1)
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 

#Imputing
from sklearn.preprocessing import Imputer 

#Impute with mean all 0 readings
fill_0 = Imputer(missing_values=0, strategy="mean")

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense

#Initialize
classifier_nn = Sequential()

#Input Layer
classifier_nn.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim =8))

#Hidden Layer
classifier_nn.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

#output Layer
classifier_nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier_nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_nn.fit(X_train, y_train, batch_size = 8, epochs = 75)


y_pred = classifier_nn.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)














