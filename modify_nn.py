from nn import NeuralNetwork, Neuron


def increase_output_neurons(old_nn: NeuralNetwork, num_new_outputs: int) -> NeuralNetwork:
    """
    Increase the number of output neurons in the neural network by num_new_outputs.

    Args: 
    old_nn (nn.NeuralNetwork): The old neural network.
    num_new_outputs (int): The number of new output neurons to add.

    Returns:
    nn.NeuralNetwork: The new neural network with the increased number of output neurons.
    """
    new_model: NeuralNetwork = NeuralNetwork(old_nn.num_inputs,
                                             old_nn.num_hidden_layers,
                                             old_nn.num_hidden_layer_neurons,
                                             num_new_outputs, old_nn.l2_lambda)

    # Copy weights from old model to new model
    for old_layer, new_layer in zip(old_nn.layers[:-1], new_model.layers[:-1]):
        new_layer.neurons = [Neuron(new_layer.num_inputs, new_layer.activation_function,
                                    new_model.l2_lambda) for _ in range(len(old_layer.neurons))]
        for old_neuron, new_neuron in zip(old_layer.neurons, new_layer.neurons):
            new_neuron.weights = old_neuron.weights.copy()
            new_neuron.bias = old_neuron.bias

    # Initialize new output layer
    new_output_layer = new_model.layers[-1]
    old_output_layer = old_nn.layers[-1]
    new_output_layer.neurons = [Neuron(new_output_layer.num_inputs,
                                       new_output_layer.activation_function,
                                       new_model.l2_lambda) for _ in range(num_new_outputs)]

    # Set weights for new neurons to be random or pre-initialized
    for neuron in new_output_layer.neurons[:len(old_output_layer.neurons)]:
        old_neuron = old_output_layer.neurons.pop(0)
        neuron.weights = old_neuron.weights.copy()
        neuron.bias = old_neuron.bias

    return new_model
