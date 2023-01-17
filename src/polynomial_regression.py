import numpy
import sys
import csv
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, PolynomialFeatures

df = pandas.read_csv(sys.argv[1])
df = df.fillna(0)

X = scale(df[["p90_len"]])
y = scale(df["mt_mo_ratio"])

# Create the transformer
poly_transformer = PolynomialFeatures(degree=4, include_bias=False)

# Transform the features
X_poly = poly_transformer.fit_transform(X)

# Create the linear regression model
poly_model = LinearRegression()

# Fit the model to the transformed features
poly_model.fit(X_poly, y)
# Make predictions using the fitted model
y_pred = poly_model.predict(X_poly)

plt.scatter(X, y, s=10)
plt.plot(X, y_pred, color='r')
plt.show()
