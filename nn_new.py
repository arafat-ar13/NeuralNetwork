import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

import random
from functions import *

class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = np.random.randn(num_inputs) * np.sqrt(2. / num_inputs)  # He initialization
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.d_weights = None
        self.d_bias = None
        self.delta = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.bias

        if self.activation_function == "relu":
            self.output = relu(self.output)

        return self.output
    
    def backward(self, delta, learning_rate):
        if self.activation_function == "relu":
            delta *= relu_derivative(self.output)
        
        self.d_weights = delta * self.inputs
        self.d_bias = delta
        
        self.delta = delta
        return np.dot(delta, self.weights)
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])

        if self.activation_function == "softmax":
            self.outputs = softmax(self.outputs)

        return self.outputs
    
    def backward(self, delta, learning_rate):
        next_delta = np.zeros(self.num_inputs)
        for i, neuron in enumerate(self.neurons):
            next_delta += neuron.backward(delta[i], learning_rate)
        return next_delta

    def update(self, learning_rate):
        for neuron in self.neurons:
            neuron.update(learning_rate)

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_output_layer_neurons):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_output_layer_neurons = num_output_layer_neurons
        self.layers = []
        input_size = num_inputs

        for _ in range(num_hidden_layers):
            layer = Layer(input_size, self.num_hidden_layer_neurons, activation_function="relu")
            self.layers.append(layer)
            input_size = self.num_hidden_layer_neurons

        output_layer = Layer(num_inputs=input_size, num_neurons=num_output_layer_neurons, activation_function="softmax")
        self.layers.append(output_layer)

    def print_info(self):
        """Print the weights and biases"""
        for i, layer in enumerate(self.layers):
            print(f"For layer {i+1}")
            print(f"Number of neurons in layer: {len(layer.neurons)}")
            for j, neuron in enumerate(layer.neurons):
                print(f"Layer {i+1}, neuron {j+1}")
                print(f"Neuron Weights: {neuron.weights}. Bias: {neuron.bias}")
            print("\n")

    def forward(self, inputs):
        layer_inputs = inputs
        for layer in self.layers:
            layer_inputs = layer.forward(layer_inputs)
        return layer_inputs

    def calc_loss_delta(self, predicted_outputs, actual_outputs):
        return predicted_outputs - actual_outputs

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(X.shape[0]):
                inputs = X[i]
                expected_output = y[i]
                
                # Forward pass
                predicted_output = self.forward(inputs)
                
                # Calculate loss
                loss = categorical_cross_entropy(predicted_output, expected_output)
                total_loss += loss
                
                # Calculate loss gradient (delta for the output layer)
                loss_delta = self.calc_loss_delta(predicted_output, expected_output)

                # Backward pass
                delta = loss_delta
                for layer in reversed(self.layers):
                    delta = layer.backward(delta, learning_rate)
                    layer.update(learning_rate)
            
            average_loss = total_loss / X.shape[0]
            print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')


# Load the MNIST dataset from files
mndata = MNIST('data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Convert lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(type(y_test))

# Normalize the data
X_train = X_train / 255.0
# X_test = X_test / 255.0

# scaler = StandardScaler()
# X_train_normalized = scaler.fit_transform(X_train)
# X_test_normalized = scaler.transform(X_test)

# Flatten images for the neural network
# X_train_flat = X_train.reshape(-1, 28*28)
# X_test_flat = X_test.reshape(-1, 28*28)

# # print(X_train_flat.shape)

# # One-hot encode the labels
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

# # Flatten the images
# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

# print(X_test.shape)

nn = NeuralNetwork(num_inputs=28*28, num_hidden_layers=3, num_hidden_layer_neurons=11, num_output_layer_neurons=10)
# nn.train(X_train, y_train_encoded, epochs=5, learning_rate=0.01)


# print(nn.forward(X_test[55]))


print(mndata.display(X_test[801]))
X = np.array([
    [1200, 1.6, 4],
    [1500, 2.0, 4],
    [1800, 2.5, 4],
    [2200, 3.0, 2],
    [2500, 3.5, 2],
    [2100, 2.8, 2],
    [800, 0.8, 2],
    [950, 1.0, 2],
    [1100, 1.2, 2],
    [1300, 1.4, 4]
])

# one hot encoding

# [car, truck, bike]

y = np.array([
    [1, 0, 0],  # Car
    [1, 0, 0],  # Car
    [1, 0, 0],  # Car
    [0, 1, 0],  # Truck
    [0, 1, 0],  # Truck
    [0, 1, 0],  # Truck
    [0, 0, 1],  # Bike
    [0, 0, 1],  # Bike
    [0, 0, 1],  # Bike
    [1, 0, 0]   # Car
])

# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)

# nn = NeuralNetwork(num_inputs=3, num_hidden_layers=2, num_hidden_layer_neurons=4, num_output_layer_neurons=3)

# # print(X)
# # # Use the normalized data for training
# nn.train(X_normalized, y, epochs=1000, learning_rate=0.1)

# Predict with normalized input
# print(nn.forward(np.array([[1100, 1.2, 2]])))

# new = np.array([3000, 3.0, 2])

np.set_printoptions(suppress=True, precision=10)


# # print("\n")
# print(nn.forward(scaler.transform([new])[0]))

# print(scaler.transform([new])[0])


# Function to generate synthetic data
# def generate_data(num_samples):
#     X = []
#     y = []
#     for _ in range(num_samples):
#         # Randomly generate features
#         weight = np.random.uniform(500, 3000)  # Weight in kg
#         engine_size = np.random.uniform(0.5, 4.0)  # Engine size in liters
#         num_doors = np.random.choice([2, 4])  # Number of doors

#         # Determine the class based on some arbitrary rules
#         if weight > 2000:
#             label = [0, 1, 0]  # Truck
#         elif engine_size < 1.5:
#             label = [0, 0, 1]  # Bike
#         else:
#             label = [1, 0, 0]  # Car

#         X.append([weight, engine_size, num_doors])
#         y.append(label)

#     return np.array(X), np.array(y)

# # Generate synthetic data
# num_samples = 1000
# X, y = generate_data(num_samples)

# # Normalize the data
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.35, random_state=42)

# # Initialize and train the neural network
# nn = NeuralNetwork(num_inputs=3, num_hidden_layers=2, num_hidden_layer_neurons=4, num_output_layer_neurons=3)
# nn.train(X_train, y_train, epochs=100, learning_rate=0.01)

# # Test the neural network
# def test_accuracy(nn, X_test, y_test):
#     correct_predictions = 0
#     for i in range(X_test.shape[0]):
#         predicted_output = nn.forward(X_test[i])
#         predicted_label = np.argmax(predicted_output)
#         actual_label = np.argmax(y_test[i])
#         if predicted_label == actual_label:
#             correct_predictions += 1
#     accuracy = correct_predictions / X_test.shape[0]
#     return accuracy

# accuracy = test_accuracy(nn, X_test, y_test)
# print(f'Accuracy: {accuracy:.4f}')
