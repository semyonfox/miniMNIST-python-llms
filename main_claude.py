from dataclasses import dataclass
import numpy as np

# Neural Network Configuration - Optimized
INPUT_SIZE = 784
HIDDEN_SIZE = 512  # Increased from 256 for more capacity
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001  # Balanced learning rate
EPOCHS = 100  # Increased from 50
BATCH_SIZE = 128  # Increased from 60 for more stable gradients
MOMENTUM = 0.9  # Added momentum for better optimization
WEIGHT_DECAY = 0.0001  # L2 regularization


@dataclass
class Layer:
    """
    Represents a layer in a feed-forward neural network with momentum

    :ivar weights: The weight of each connection between a layer input and a layer output
    :type weights: np.ndarray

    :ivar biases: The bias added to each output of the layer
    :type biases: np.ndarray

    :ivar weight_velocity: Momentum term for weight updates
    :type weight_velocity: np.ndarray

    :ivar bias_velocity: Momentum term for bias updates
    :type bias_velocity: np.ndarray
    """
    weights: np.ndarray
    biases: np.ndarray
    weight_velocity: np.ndarray
    bias_velocity: np.ndarray

    @classmethod
    def initialize(cls, inputs: int, outputs: int):
        """
        Returns a new Layer with the specified number of inputs and outputs.
        The biases are initialised to zero, and the weights are initialised using the Kaiming Initialisation method

        :param inputs: The number of inputs to the layer
        :param outputs: The number of outputs from the layer
        :returns: A new Layer with weights and biases
        :rtype: Layer
        """
        stddev = np.sqrt(2.0 / inputs)
        weights = np.random.normal(loc=0.0, scale=stddev, size=(inputs, outputs))
        biases = np.zeros(outputs)

        # Initialize momentum terms
        weight_velocity = np.zeros_like(weights)
        bias_velocity = np.zeros_like(biases)

        return cls(weights, biases, weight_velocity, bias_velocity)


@dataclass
class Network:
    """
    A Neural Network with one hidden layer and one output layer
    :ivar hidden: The Hidden Layer
    :ivar output: The Output Layer
    :type hidden: Layer
    :type output: Layer
    """
    hidden: Layer
    output: Layer


@dataclass
class InputData:
    """
    Input data for the Neural Network, consisting of images, labels, and the number of images in the dataset
    :ivar images: An array of images
    :ivar labels: An array of labels
    :ivar image_count: The number of images in `images`
    :type images: np.ndarray
    :type labels: np.ndarray
    :type image_count: int
    """
    images: np.ndarray
    labels: np.ndarray
    image_count: int


def load_mnist_images(filename: str) -> (int, np.ndarray):
    """
    Loads an MNIST image dataset (can be test or train)
    Transforms each pixel in each image into a float between 0.0 and 1.0
    :param filename: File or File Path containing MNIST images
    :type filename: str

    :returns A tuple containing the number of images as an int and the image array as an np.ndarray
    :rtype: (int, np.ndarray)
    """
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return num_images, images.astype(np.float32) / 255.0


def load_mnist_labels(filename: str) -> (int, np.ndarray):
    """
    Loads an MNIST label dataset (can be test or train)
    All labels are loaded as 8-bit unsigned integers

    :param filename: File or File Path containing MNIST labels

    :returns A tuple containing the number of labels as an int and the label array as an np.ndarray
    :rtype: (int, np.ndarray)
    """
    with open(filename, 'rb') as f:
        _, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return num_labels, labels


def forward_propagation(layer: Layer, inputs: np.ndarray) -> np.ndarray:
    """
    Performs forward propagation on the specified Layer with the provided inputs and returns the pre-activation outputs of the layer.

    :param layer: The Layer to be acted upon
    :param inputs: The inputs to the layer
    :type layer: Layer
    :type inputs: np.ndarray

    :returns A 1-D array containing the pre-activation outputs of the Layer
    :rtype: np.ndarray
    """
    return np.dot(inputs, layer.weights) + layer.biases


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Applies the softmax activation function to the input array, and normalises the output.

    :param x: The pre-activation output array of a neural network layer
    :type x: np.ndarray

    :returns: A probability distribution containing the normalised probabilities of the input values
    :rtype: np.ndarray
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit (ReLU) activation function to the input array.
    """
    return np.maximum(0, x)


def backward_propagation(layer: Layer, input: np.ndarray, output_grad: np.ndarray,
                         learning_rate: float, momentum: float, weight_decay: float) -> np.ndarray:
    """
    Performs backward propagation with momentum on the specified Layer.

    :param layer: The layer to back propagate
    :type layer: Layer

    :param input: The layer inputs which led to the output gradient
    :type input: np.ndarray

    :param output_grad: The output gradient for the layer given the inputs specified in `input`
    :type output_grad: np.ndarray

    :param learning_rate: The learning rate of the neural network
    :type learning_rate: float

    :param momentum: The momentum coefficient
    :type momentum: float

    :param weight_decay: L2 regularization coefficient
    :type weight_decay: float

    :returns: The loss gradient with respect to the layer inputs
    :rtype: np.ndarray
    """
    # Calculate gradients with L2 regularization
    weight_grad = np.dot(input.transpose(), output_grad) + weight_decay * layer.weights
    bias_grad = np.sum(output_grad, axis=0)

    # Update velocities with momentum
    layer.weight_velocity = momentum * layer.weight_velocity + learning_rate * weight_grad
    layer.bias_velocity = momentum * layer.bias_velocity + learning_rate * bias_grad

    # Update weights and biases
    layer.weights -= layer.weight_velocity
    layer.biases -= layer.bias_velocity

    # Calculate and return input gradient
    input_grad = np.dot(output_grad, layer.weights.transpose())
    return input_grad


def train(net: Network, input: np.ndarray, label: np.ndarray,
          learning_rate: float, momentum: float, weight_decay: float):
    """
    Trains the provided neural network with the given input array and label array.

    :param net: The network to be trained
    :type net: Network

    :param input: The array of inputs to use for training
    :type input: np.ndarray

    :param label: The array of true labels to use for training
    :type label: np.ndarray

    :param learning_rate: The learning rate of the network
    :type learning_rate: float

    :param momentum: The momentum coefficient
    :type momentum: float

    :param weight_decay: L2 regularization coefficient
    :type weight_decay: float
    """
    # Forward pass
    hidden_output = relu(forward_propagation(net.hidden, input))
    final_output = softmax(forward_propagation(net.output, hidden_output))

    # Compute output gradient
    output_grad = final_output - label

    # Backward pass with momentum and weight decay
    hidden_grad = backward_propagation(net.output, hidden_output, output_grad,
                                       learning_rate, momentum, weight_decay)
    hidden_grad *= (hidden_output > 0)
    backward_propagation(net.hidden, input, hidden_grad, learning_rate, momentum, weight_decay)


def predict(net: Network, input: np.ndarray) -> np.ndarray:
    """
    Generate predictions using the network.

    :param net: The network to generate a prediction with
    :type net: Network

    :param input: The input to the neural network
    :type input: np.ndarray

    :returns: An array containing the predictions for the provided inputs
    :rtype: np.ndarray
    """
    hidden_output = relu(forward_propagation(net.hidden, input))
    final_output = softmax(forward_propagation(net.output, hidden_output))
    return np.argmax(final_output, axis=1)


def main():
    # Load training data
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")
    training_data = InputData(training_images, training_labels, num_images)

    # Load test data
    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")
    test_data = InputData(test_images, test_labels, num_test_images)

    # Initialize network
    net = Network(
        hidden=Layer.initialize(inputs=INPUT_SIZE, outputs=HIDDEN_SIZE),
        output=Layer.initialize(inputs=HIDDEN_SIZE, outputs=OUTPUT_SIZE)
    )

    best_accuracy = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        # Shuffle training data for better generalization
        indices = np.random.permutation(num_images)
        shuffled_images = training_data.images[indices]
        shuffled_labels = training_data.labels[indices]

        total_loss = 0.0

        # Batch training
        for i in range(0, num_images, BATCH_SIZE):
            batch_images = shuffled_images[i:i + BATCH_SIZE]
            batch_labels = shuffled_labels[i:i + BATCH_SIZE]

            one_hot_labels = np.eye(OUTPUT_SIZE)[batch_labels]
            train(net, batch_images, one_hot_labels, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)

            # Calculate loss
            hidden_output = relu(forward_propagation(net.hidden, batch_images))
            final_output = softmax(forward_propagation(net.output, hidden_output))

            log_probs = np.log(np.clip(final_output[range(len(batch_labels)), batch_labels], 1e-10, 1.0))
            total_loss -= np.sum(log_probs)

        # Evaluate on test set
        predictions = predict(net, test_data.images)
        accuracy = np.mean(predictions == test_data.labels)

        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        print(f"Epoch {epoch + 1}/{EPOCHS}, Accuracy: {accuracy * 100:.2f}%, "
              f"Avg Loss: {total_loss / num_images:.4f}, Best: {best_accuracy * 100:.2f}%")

    print(f"\nFinal Best Accuracy: {best_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()