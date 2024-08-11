from nn import *

from functions import load_model
from modify_nn import increase_output_neurons
from PIL import Image

import random

# # Loading in the data
# X_train, y_train, X_test, y_test = load_mnist_data("data")

# # One-hot encoding the labels
# encoder = OneHotEncoder()
# y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
# y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

# # Normalizing the data
# scaler = StandardScaler()
# X_train_normalized = X_train / 255.0
# # X_test_normalized = scaler.transform(X_test)

# # Creating constants for the neural network
# NUM_INPUTS = X_train_normalized.shape[1]
# NUM_HIDDEN_LAYERS = 3
# NUM_HIDDEN_LAYER_NEURONS = 11
# NUM_OUTPUT_LAYER_NEURONS = y_train_encoded.shape[1]

# print(y_train_encoded)
# Creating the neural network
# nn = NeuralNetwork(num_inputs=NUM_INPUTS, num_hidden_layers=NUM_HIDDEN_LAYERS,
#                    num_hidden_layer_neurons=NUM_HIDDEN_LAYER_NEURONS,
#                    num_output_layer_neurons=NUM_OUTPUT_LAYER_NEURONS,)

# # Training the neural network
# nn.train(X_train_normalized, y_train_encoded, epochs=5, learning_rate=0.01, max_grad_norm=2.0)
# nn.save_model("mnist_model")

# nn: NeuralNetwork = load_model("quickdraw_model")

# # nn.print_info()
# nn.l2_lambda = 0.001

# new_nn = increase_output_neurons(nn, 7)

class_names = ["bees", "cats", "bananas", "flowers", "hand", "ice cream", "microphone"]

# data_dir = "data/quickdraw"

# data = []
# labels = []


# # # iterate over all the images and save them and their labels
# for name in class_names:
#     counter = 0
#     images = os.listdir(f"{data_dir}/{name}")
#     if name in ["bees", "cats", "bananas"]:
#         random.shuffle(images)
    
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
#         counter += 1

#         if name in ["bees", "cats", "bananas"]:
#             if counter == 500:
#                 break

# # print(data)
# # print(labels)

# data = np.array(data)
# data = np.vstack(data)
# labels = np.array(labels)

# # print(data.shape)
# # print(labels.shape)
# new_nn.train(data, labels, 150, 0.001, 32)
# new_nn.save_model("new_quickdraw")

# new_nn.print_info()
# nn.print_info()

# new_nn = increase_output_neurons(nn, 4)
# print(nn.forward(X_test[899]))

# show the image mnist
# print(mndata.display(X_test[899]))
# print(X_test[899].shape)

# class_names = ["bees", "cats", "bananas", "flower"]

nn: NeuralNetwork = load_model("new_quickdraw")

new_image = Image.open("image.png")
#
new_image = new_image.resize((28, 28)).convert("L")
new_image = np.array(new_image) / 255
new_image = new_image.reshape(-1)
# new_image = (np.array(new_image.convert('L'))).reshape(-1) / 255

np.set_printoptions(suppress=True, precision=10)
prediction = nn.forward(new_image)
print(class_names[list(prediction).index(np.max(prediction))])
print(prediction)