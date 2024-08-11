"""
A Neural Network made using Numpy. Features:
- Uses Rectified Linear Unit (ReLU) Activation function for hidden layers.
- Uses Softmax Activation function for output layer.
- Has back propagation
"""

import os
import pickle
import numpy as np
from functions import categorical_cross_entropy, load_model, relu, relu_derivative, softmax

class Neuron:
    """A Single Neuron"""
    def __init__(self, num_inputs: int, activation_function: str, l2_lambda: float = 0.0):
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = np.random.randn(num_inputs) * np.sqrt(2. / num_inputs)  # He initialization
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.d_weights = None
        self.d_bias = None
        self.delta = 0
        self.l2_lambda = l2_lambda  # Regularization parameter

    def forward(self, inputs):
        """ The feedforward algorithm """
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.bias

        if self.activation_function == "relu":
            self.output = relu(self.output)
        # elif self.activation_function == "softmax":
        #     raise ValueError("Softmax should not be used in individual neurons' activation")

        return self.output

    def backward(self, delta):
        """ Back propagation on a single neuron """
        if self.activation_function == "relu":
            delta *= relu_derivative(self.output)
        # elif self.activation_function == "softmax":
        #     raise ValueError("Softmax should not be used in individual neurons' activation")

        self.d_weights = delta * self.inputs + self.l2_lambda * self.weights
        self.d_bias = delta

        self.delta = delta
        return np.dot(delta, self.weights)

    def update(self, learning_rate, max_grad_norm=None):
        """ Update the weights and biases of the neurons with gradient clipping """
        if max_grad_norm is not None:
            # Compute the gradient norm
            norm = np.sqrt(np.sum(self.d_weights**2) + self.d_bias**2)
            if norm > max_grad_norm:
                # Clip gradients
                scale = max_grad_norm / norm
                self.d_weights *= scale
                self.d_bias *= scale

        # Update weights and biases
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

class Layer:
    """A Single Layer in the Neural Network. Houses the neurons."""
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: str, l2_lambda: float = 0.0):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.neurons = [Neuron(num_inputs, activation_function, l2_lambda) for _ in range(num_neurons)]
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """The feedforward that calls the forward method on all the neurons in the layer"""
        self.inputs = inputs
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])

        if self.activation_function == "softmax":
            self.outputs = softmax(self.outputs)

        return self.outputs

    def backward(self, delta):
        """The back prop algorithm that back propagates through each neuron in the layer"""
        next_delta = np.zeros(self.num_inputs)
        for i, neuron in enumerate(self.neurons):
            next_delta += neuron.backward(delta[i])
        return next_delta

    def update(self, learning_rate, max_grad_norm=None):
        """ Updates the weights and biases of all the neurons in the layer """
        for neuron in self.neurons:
            neuron.update(learning_rate, max_grad_norm)

class NeuralNetwork:
    """ The Neural Network structure. Contains all the Layers and inputs"""
    MODEL_SAVE_PATH = "models/"

    def __init__(self, num_inputs: int, num_hidden_layers: int, num_hidden_layer_neurons: int,
                 num_output_layer_neurons: int, l2_lambda: float = 0.0):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_output_layer_neurons = num_output_layer_neurons
        self.l2_lambda = l2_lambda
        self.layers = []
        input_size = num_inputs

        for _ in range(num_hidden_layers):
            layer = Layer(input_size, self.num_hidden_layer_neurons, activation_function="relu", 
                          l2_lambda=l2_lambda)
            self.layers.append(layer)
            input_size = self.num_hidden_layer_neurons

        output_layer = Layer(num_inputs=input_size, num_neurons=num_output_layer_neurons, 
                             activation_function="softmax", l2_lambda=self.l2_lambda)
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

    def save_model(self, filename):
        """Method to save the model to a file"""
        if not os.path.exists(self.MODEL_SAVE_PATH):
            os.makedirs(self.MODEL_SAVE_PATH)

        filepath = os.path.join(self.MODEL_SAVE_PATH, f"{filename}.pkl")
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved successfully to {filepath}!")

    def forward(self, inputs):
        """Feedforward that propagates to all the layers in the network"""
        layer_inputs = inputs
        for layer in self.layers:
            layer_inputs = layer.forward(layer_inputs)
        return layer_inputs

    def train(self, X, y, epochs, learning_rate, batch_size=32, max_grad_norm=None):
        """Train the model for a certain number of epochs using mini-batch gradient descent"""
        num_samples = X.shape[0]
        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(num_samples)
            np.random.shuffle(indices)  # Shuffle data for each epoch

            # Mini-batch training
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                batch_loss = 0
                for i, _ in enumerate(X_batch):
                    inputs = X_batch[i]
                    expected_output = y_batch[i]

                    # Forward pass
                    predicted_output = self.forward(inputs)

                    # Calculate loss
                    loss = categorical_cross_entropy(predicted_output, expected_output)
                    batch_loss += loss

                    # Calculate loss gradient (delta for the output layer)
                    loss_delta = predicted_output - expected_output

                    # Backward pass
                    delta = loss_delta
                    for layer in reversed(self.layers):
                        delta = layer.backward(delta)
                        layer.update(learning_rate, max_grad_norm)

                total_loss += batch_loss / len(X_batch)

            average_loss = total_loss / (num_samples / batch_size)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')

# class_names = ["bees", "cats", "bananas"]

# data_dir = "data/quickdraw/"

# data = []
# labels = []

# for name in class_names:
#     images = os.listdir(f"{data_dir}/{name}")

#     # iterate over all the images and save them and their labels
#     for image in images:
#         img = Image.open(f"{data_dir}/{name}/{image}")

#         # greyscale it
#         img = img.convert('L')

#         # normalize it
#         img = np.array(img) / 255

#         # reshape it
#         img = img.reshape(1, -1)

#         data.append(img)

#         label = np.zeros(len(class_names))
#         label[class_names.index(name)] = 1
#         labels.append(label)

# data = np.array(data)
# data = np.vstack(data)
# labels = np.array(labels)

# nn = NeuralNetwork(data.shape[1], 5, 11, labels.shape[1], 0.001)
# nn.train(data, labels, 380, 0.001, batch_size=32)
# nn.save_model("quickdraw_model")
# nn: NeuralNetwork = load_model("quickdraw_model")

# # # new_image = Image.open("data/quickdraw/bananas/bananas802.png")
# # # new_image = Image.open("data/banana.png")
# new_image = Image.open("image.png")
# #
# new_image = new_image.resize((28, 28)).convert("L")
# new_image = np.array(new_image) / 255
# new_image = new_image.reshape(-1)
# # new_image = (np.array(new_image.convert('L'))).reshape(-1) / 255

# np.set_printoptions(suppress=True, precision=10)
# prediction = nn.forward(new_image)
# print(class_names[list(prediction).index(np.max(prediction))])
# print(prediction)

# nn.print_info()

# print(new_image.shape)

# def preprocess_image(image_path):
#     # Open the image
#     img = Image.open(image_path).convert('L')  # Convert to grayscale

#     # Resize the image to fit within 255x255 while maintaining aspect ratio
#     img.thumbnail((255, 255), Image.Resampling.LANCZOS)

#     # Create a new 255x255 canvas
#     canvas = Image.new('L', (255, 255), color=255)  # White background

#     # Calculate position to paste the resized image
#     x_offset = (255 - img.width) // 2
#     y_offset = (255 - img.height) // 2

#     # Paste the resized image onto the canvas
#     canvas.paste(img, (x_offset, y_offset))

#     # Resize the canvas image to 28x28
#     img_resized = canvas.resize((28, 28), Image.Resampling.LANCZOS)

#     # Convert image to a numpy array
#     img_array = np.array(img_resized)

#     # Flatten the image array
#     # img_flattened = img_array.flatten()

#     # Normalize the pixel values
#     img_normalized = img_array.reshape(-1) / 255.0

#     return img_normalized

# print(nn.forward(preprocess_image("cat_1.png")))