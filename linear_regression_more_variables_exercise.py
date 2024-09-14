import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Create a dataset with three variables
data = {
    'Surface' : [30, 50, 70, 100, 120],
    'Year_construction' : [2000, 1990, 1985, 2010, 2005],
    'Price' : [150000, 200000, 250000, 300000, 350000]
}

# Convert dataset to Dataframe
df = pd.DataFrame(data)

print(df)

# Definition of independent (X) and dependent (y) variables
X = df[['Surface', 'Year_construction']]  # X -> features
y = df['Price']  # y -> is the target variable I want to predict

# Create a model of linear regression
model = LinearRegression()
model.fit(X,y)  # trains the model on data. It finds the best-fitting plane (two features)

# Price prediction using the model
y_pred = model.predict(X)  # Using the trained model to predict the house prices

# View model coefficients
print(f"Coefficient for the Surface: {model.coef_[0]:.2f}")  # 2260.29
# means that for each additional square meter of surface, the house price increases by $2260.29
print(f"Coefficient for the Year of Construction: {model.coef_[1]:.2f}")  # -664.26
# means that for each additional year, the price of the house decreases by $664.26.
# This means that, for this dataset, older houses tend to be more valuable than newer ones.
print(f"Intercept: {model.intercept_:.2f}")


# Calculate mean_squared_error (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Calculate R^2
r2= r2_score(y, y_pred)
print(f"R^2 score: {r2:.2f}")