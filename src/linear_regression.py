import numpy
import sys
import csv
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pandas.read_csv(sys.argv[1])
df = df.fillna(0)

X = df[['p90_len']]
Y = df['mt_mo_ratio']

X_train, X_test, y_train, y_test = train_test_split(scale(X), scale(Y), test_size=0.2, random_state=42)

#slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

#r_squared = r_value ** 2

#print(f"slope {slope} intercept {intercept} r val {r_value} r squared {r_squared} p value {p_value} std err {std_err}")

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, predictions, color='red')
plt.show()

