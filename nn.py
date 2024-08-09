"""
A Neural Network made using Numpy. Features:
- Uses Rectified Linear Unit (ReLU) Activation function for hidden layers.
- Uses Softmax Activation function for output layer.
- Has back propagation
"""

import os
import pickle

import numpy as np

from functions import categorical_cross_entropy, relu, relu_derivative, softmax


class Neuron:
    """A Single Neuron"""
    def __init__(self, num_inputs: int, activation_function: str):
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = np.random.randn(
            num_inputs) * np.sqrt(2. / num_inputs)  # He initialization
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.d_weights = None
        self.d_bias = None
        self.delta = 0

    def forward(self, inputs):
        """ The feedforward algorithm """
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.bias

        if self.activation_function == "relu":
            self.output = relu(self.output)

        return self.output

    def backward(self, delta):
        """ Back propagation on a single neuron """
        if self.activation_function == "relu":
            delta *= relu_derivative(self.output)

        self.d_weights = delta * self.inputs
        self.d_bias = delta

        self.delta = delta
        return np.dot(delta, self.weights)

    def update(self, learning_rate):
        """ Update the weights and biases of the neurons """
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias


class Layer:
    """A Single Layer in the Neural Network. Houses the neurons."""
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: str):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.neurons = [Neuron(num_inputs, activation_function)
                        for _ in range(num_neurons)]

        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """The feedforward that calls the forward method on all the neurons in the layer"""
        self.inputs = inputs
        self.outputs = np.array([neuron.forward(inputs)
                                for neuron in self.neurons])

        if self.activation_function == "softmax":
            self.outputs = softmax(self.outputs)

        return self.outputs

    def backward(self, delta):
        """The back prop algorithm that back propagates through each neuron in the layer"""
        next_delta = np.zeros(self.num_inputs)
        for i, neuron in enumerate(self.neurons):
            next_delta += neuron.backward(delta[i])
        return next_delta

    def update(self, learning_rate):
        """ Updates the weights and biases of all the neurons in the layer """
        for neuron in self.neurons:
            neuron.update(learning_rate)


class NeuralNetwork:
    """ The Neural Network structure. Contains all the Layers and inputs"""
    MODEL_SAVE_PATH = "models/"

    def __init__(self, num_inputs: int, num_hidden_layers: int, num_hidden_layer_neurons: int,
                num_output_layer_neurons: int):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_output_layer_neurons = num_output_layer_neurons
        self.layers = []
        input_size = num_inputs

        for _ in range(num_hidden_layers):
            layer = Layer(input_size, self.num_hidden_layer_neurons,
                          activation_function="relu")
            self.layers.append(layer)
            input_size = self.num_hidden_layer_neurons

        output_layer = Layer(
        num_inputs=input_size, num_neurons=num_output_layer_neurons, activation_function="softmax")
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
        """Feedforward that propagates to all the layers in the network"""
        layer_inputs = inputs
        for layer in self.layers:
            layer_inputs = layer.forward(layer_inputs)
        return layer_inputs

    def train(self, X, y, epochs, learning_rate):
        """Train the model a certain number of epochs"""
        for epoch in range(epochs):
            total_loss = 0
            for i in range(X.shape[0]):
                inputs = X[i]
                expected_output = y[i]

                # Forward pass
                predicted_output = self.forward(inputs)

                # Calculate loss
                loss = categorical_cross_entropy(
                    predicted_output, expected_output)
                total_loss += loss

                # Calculate loss gradient (delta for the output layer)
                loss_delta = predicted_output - expected_output

                # Backward pass
                delta = loss_delta
                for layer in reversed(self.layers):
                    delta = layer.backward(delta)
                    layer.update(learning_rate)

            average_loss = total_loss / X.shape[0]
            print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')
