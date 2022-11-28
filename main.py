import numpy as np
from neural import NeuralNetwork

def main():
    """ 
    Main function of the program. 
    """
    # Creating multiple datasets
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    X_2 = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_2 = np.array([[1], [1], [0], [0]])

    X_3 = np.random.rand(100, 2)
    y_3 = np.array([[1 if x[0] > x[1] else 0] for x in X_3])

    # Creating the neural network
    nn = NeuralNetwork()

    # Training the neural network
    nn.fit(X, y)
    nn.fit(X_2, y_2)
    nn.fit(X_3, y_3)

    # Preding the output
    print(f"Prediction for the first data set: \n{nn.predict(X)}")
    print(f"\nPrediction for the second data set: \n{nn.predict(X_2)}")
    print(f"\nPrediction for the third data set: \n{nn.predict(X_3)}")

if __name__ == "__main__":
    main()