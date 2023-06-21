import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file = 'real-data.csv'
dataset = pd.read_csv(file, usecols=["age", "motivation", "interest", "block programming", "code programming", "self confidence", "learning performance"])

x = dataset[['age', 'motivation', 'interest', 'block programming', 'code programming', 'self confidence']]
y = dataset['learning performance']

#Splitting the dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

mlr = LinearRegression()  
mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
for i in zip(x,mlr.coef_):
    print(i)

y_pred_mlr= mlr.predict(x_test)
print("Prediction for test set: {}".format(y_pred_mlr))
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})

print(mlr_diff.head())


from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
plt.clf()
sns.heatmap(dataset.corr(), cmap="YlGnBu", annot = True)
title = "graphs/" + "headmap" + '.png'
plt.savefig(title)
