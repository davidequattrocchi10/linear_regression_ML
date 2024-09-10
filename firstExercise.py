import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a small sample dataset (surface vs price)
data = {
    'Surface' : [30, 50, 70, 100, 120],
    'Price' : [150000, 200000, 250000, 300000, 350000]
}

# Convert dataset to Dataframe
df = pd.DataFrame(data)

# See dataset
print(df)

# See data
# plt.scatter(df['Surface'], df['Price'])
# plt.xlabel('Surface (m^2)')
#plt.ylabel('Price ($)')
# plt.title('Surface vs Price')
# plt.show()

# Define independent variable (X) and dependent variable (y)
X = df['Surface']
y = df['Price']

# Calculate regression coefficients (m,b)
# m is the slope, b is the intercept (y=m*X + b)
m, b = np.polyfit(X, y, 1)


print(f"Formula: Price = {m:.2f} * Surface + {b:.2f}")


# Visualize of linear regression
plt.scatter(X, y)
plt.plot(X, m*X + b, color='red') # linear regression
plt.xlabel('Surface (m^2)')
plt.ylabel('Price ($)')
plt.title('Surface vs Price with Linear Regression')
plt.show()

