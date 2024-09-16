import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Function to calculate 'mean squared error' (cost function)
def compute_cost(X, y, theta):
    """
    This function computes the cost (mean squared error) for linear regression.

    Args:
    X : np.ndarray : Input features matrix
    y : np.ndarray : Target variable
    theta : np.ndarray : Parameters (weights) for the model

    Returns:
    float : The computed cost
    """
    m = len(y)  # Number of training examples
    predictions = X.dot(theta)  # Predictions using the current theta
    error = predictions - y  # Difference between predictions and actual values
    cost = (1 / (2 * m)) * np.dot(error.T, error)  # MSE cost function
    return cost


# Function for Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    This function performs gradient descent to learn the optimal parameters theta.

    Args:
    X : np.ndarray : Input features matrix
    y : np.ndarray : Target variable
    theta : np.ndarray : Initial parameters (weights)
    learning_rate : float : The step size for gradient updates
    iterations : int : Number of iterations to run gradient descent

    Returns:
    np.ndarray : The optimized theta (parameters)
    np.ndarray : The cost history during gradient descent
    """
    m = len(y)  # Number of training examples
    cost_history = np.zeros(iterations)  # To store cost at each iteration

    for i in range(iterations):
        predictions = X.dot(theta)  # Compute predictions
        error = predictions - y  # Calculate the error
        gradient = (1 / m) * X.T.dot(error)  # Compute the gradient
        theta = theta - learning_rate * gradient  # Update theta using gradient descent
        cost_history[i] = compute_cost(X, y, theta)  # Compute and store cost at iteration i

    return theta, cost_history


# Create a dataset with three variables
data = {
    'Surface': [30, 50, 70, 100, 120],  # Surface area of houses
    'Year_construction': [2000, 1990, 1985, 2010, 2005],  # Year of construction
    'Price': [150000, 200000, 250000, 300000, 350000]  # House prices
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Define the independent (X) and dependent (y) variables
X = df[['Surface', 'Year_construction']].values  # Input features: Surface and Year_construction
y = df['Price'].values  # Target variable: House prices

# Feature scaling (only on the input features, not the intercept)
scaler = StandardScaler()  # Initialize the standard scaler
X_scaled = scaler.fit_transform(X)  # Scale the input features

# Add a column of 1s for the intercept (bias term) after scaling
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # Concatenate bias column with scaled features

# Initialize theta (parameters) to zeros
theta = np.zeros(X_scaled.shape[1])  # Initialize theta with zeros (size of theta = number of features + 1 for bias)

# Set parameters for Gradient Descent
learning_rate = 0.001  # The step size for gradient descent
iterations = 1000  # Number of iterations

# Optimization of parameters using Gradient Descent
theta_optimized, cost_history = gradient_descent(X_scaled, y, theta, learning_rate, iterations)

# Output the optimized parameters (theta)
print(f"Optimization of parameters (theta): {theta_optimized}")

# Plotting the cost function history to see convergence
plt.plot(range(iterations), cost_history)  # Plot the cost history over iterations
plt.xlabel('Iterations')  # Label for the x-axis
plt.ylabel('Cost (MSE)')  # Label for the y-axis
plt.title('Convergence of the gradient descent')  # Title of the plot
plt.show()  # Display the plot
