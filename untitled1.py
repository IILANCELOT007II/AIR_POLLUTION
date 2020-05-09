import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, 1:13].values
Y_train = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1= LabelEncoder()
X_train[:, 0] = labelencoder_X_1.fit_transform(X_train[:, 0])
labelencoder_X_2= LabelEncoder()
X_train[:, 10] = labelencoder_X_2.fit_transform(X_train[:, 10])


df = pd.read_csv('test.csv')
X_test = df.iloc[:, 1:13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_3= LabelEncoder()
X_test[:, 0] = labelencoder_X_3.fit_transform(X_test[:, 0])
labelencoder_X_4= LabelEncoder()
X_test[:, 10] = labelencoder_X_4.fit_transform(X_test[:, 10])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

for i in range(0,14454):
    y_pred[i] = round(y_pred[i])
