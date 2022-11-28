# Importing required libraries
import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

class NeuralNetwork:
    def __init__(self) -> None:
        """ 
        Initialize the neural network
        """
        self.w1 = None
        self.w2 = None

    def fit(self, X, y, learning_rate = 0.2, epochs = 4000) -> None:
        """ Fit the neural network to the data X and y 
                X: input data
                y: output data
                learning_rate: learning rate
                epochs: number of epochs
        """
        self.w1 = truncnorm.rvs(-1, 1, size=(X.shape[1], 4))
        self.w2 = truncnorm.rvs(-1, 1, size=(4, 1))

        for _ in range(epochs):
            # Forward propagation
            z1 = X @ self.w1
            a1 = activation_function(z1)

            z2 = a1 @ self.w2
            y_hat = activation_function(z2)

            # Backpropagation
            delta2 = (y_hat - y) * y_hat * (1 - y_hat)
            delta1 = delta2 @ self.w2.T * a1 * (1 - a1)

            # Updating weights
            self.w2 -= learning_rate * a1.T @ delta2
            self.w1 -= learning_rate * X.T @ delta1

    def predict(self, X) -> np.ndarray:
        """ 
        Predict the output for the input data X
            X: input datas
        """
        z1 = X @ self.w1
        a1 = activation_function(z1)

        z2 = a1 @ self.w2
        y_hat = activation_function(z2)
        
        return y_hat