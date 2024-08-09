from sklearn.preprocessing import OneHotEncoder, StandardScaler

from functions import load_mnist_data, load_model
from nn import NeuralNetwork

# Loading in the data
X_train, y_train, X_test, y_test = load_mnist_data("data")

# One-hot encoding the labels
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

# Normalizing the data
scaler = StandardScaler()
X_train_normalized = X_train / 255.0
# X_test_normalized = scaler.transform(X_test)

# Creating constants for the neural network
NUM_INPUTS = X_train_normalized.shape[1]
NUM_HIDDEN_LAYERS = 3
NUM_HIDDEN_LAYER_NEURONS = 11
NUM_OUTPUT_LAYER_NEURONS = y_train_encoded.shape[1]

# Creating the neural network
# nn = NeuralNetwork(num_inputs=NUM_INPUTS, num_hidden_layers=NUM_HIDDEN_LAYERS,
#                    num_hidden_layer_neurons=NUM_HIDDEN_LAYER_NEURONS,
#                    num_output_layer_neurons=NUM_OUTPUT_LAYER_NEURONS)

# Training the neural network
# nn.train(X_train_normalized, y_train_encoded, epochs=5, learning_rate=0.01)
# nn.save_model("mnist_model")

nn: NeuralNetwork = load_model("mnist_model")
print(nn.forward(X_test[899]))