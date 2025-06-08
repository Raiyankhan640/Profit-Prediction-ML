import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
# print(df.shape)
# print(df.isnull().sum())

x_values = df.drop(['Profit'], axis=1)
# print(x_values.head(10))
y_values = df['Profit']

# convert the column area containing string to a categorical colum
city = pd.get_dummies(x_values['Area'], drop_first=True).astype(int)
# print(city)
x_values = x_values.drop(['Area'], axis=1)
x_values = pd.concat([x_values, city], axis=1)
# print(x_values)
x = x_values
y = y_values

# plt.plot(x_values['Marketing Spend'], y_values)
# plt.show()

#spliting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
y_pred = regressor.predict(xtest)

print(ytest)
print(y_pred)

from sklearn.metrics import r2_score
score = r2_score(ytest, y_pred)
print(score*100)

score = regressor.score(xtest, ytest)
print("Accuracy is: ", score*100)

