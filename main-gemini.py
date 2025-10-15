from dataclasses import dataclass, field
import numpy as np

# --- Optimised Neural Network Configuration ---
# Architecture
INPUT_SIZE = 784
HIDDEN_SIZE_1 = 512  # Increased neurons for more capacity
HIDDEN_SIZE_2 = 256  # Added a second hidden layer
OUTPUT_SIZE = 10

# Training Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 25  # Fewer epochs needed with a better optimizer
BATCH_SIZE = 64  # Common batch size, power of 2

# Adam Optimizer Parameters
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8

# L2 Regularization (Weight Decay)
L2_LAMBDA = 1e-4

# Learning Rate Decay
LR_DECAY = 0.95  # Decay learning rate by 5% each epoch


@dataclass
class Layer:
    """Represents a layer, now with state for the Adam optimizer."""
    weights: np.ndarray
    biases: np.ndarray
    # Adam optimizer state variables
    m_weights: np.ndarray = field(init=False)
    v_weights: np.ndarray = field(init=False)
    m_biases: np.ndarray = field(init=False)
    v_biases: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize optimizer states to zero arrays."""
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    @classmethod
    def initialize(cls, inputs: int, outputs: int):
        """Initializes a layer using Kaiming (He) Initialization."""
        stddev = np.sqrt(2.0 / inputs)
        weights = np.random.normal(loc=0.0, scale=stddev, size=(inputs, outputs))
        biases = np.zeros(outputs)
        return cls(weights, biases)


@dataclass
class Network:
    """A deeper Neural Network with two hidden layers."""
    hidden1: Layer
    hidden2: Layer
    output: Layer


@dataclass
class InputData:
    """Input data for the Neural Network."""
    images: np.ndarray
    labels: np.ndarray
    image_count: int


# --- Data Loading (Unchanged) ---
def load_mnist_images(filename: str) -> (int, np.ndarray):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return num_images, images.astype(np.float32) / 255.0


def load_mnist_labels(filename: str) -> (int, np.ndarray):
    with open(filename, 'rb') as f:
        _, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return num_labels, labels


# --- Core Functions (Updated) ---
def forward_propagation(layer: Layer, inputs: np.ndarray) -> np.ndarray:
    """Performs forward propagation on a layer."""
    return np.dot(inputs, layer.weights) + layer.biases


def relu(x: np.ndarray) -> np.ndarray:
    """Applies the ReLU activation function."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Applies the softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def adam_optimizer_update(layer: Layer, grad_weights: np.ndarray, grad_biases: np.ndarray, t: int,
                          learning_rate: float):
    """
    Updates layer weights and biases using the Adam optimization algorithm.
    Adam adapts the learning rate for each parameter, improving convergence.
    """
    # Update biased first moment estimate
    layer.m_weights = BETA1 * layer.m_weights + (1 - BETA1) * grad_weights
    layer.m_biases = BETA1 * layer.m_biases + (1 - BETA1) * grad_biases

    # Update biased second raw moment estimate
    layer.v_weights = BETA2 * layer.v_weights + (1 - BETA2) * (grad_weights ** 2)
    layer.v_biases = BETA2 * layer.v_biases + (1 - BETA2) * (grad_biases ** 2)

    # Compute bias-corrected first moment estimate
    m_weights_hat = layer.m_weights / (1 - BETA1 ** t)
    m_biases_hat = layer.m_biases / (1 - BETA1 ** t)

    # Compute bias-corrected second raw moment estimate
    v_weights_hat = layer.v_weights / (1 - BETA2 ** t)
    v_biases_hat = layer.v_biases / (1 - BETA2 ** t)

    # Update weights and biases
    layer.weights -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + EPSILON)
    layer.biases -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + EPSILON)


def backward_propagation(layer: Layer, layer_input: np.ndarray, output_grad: np.ndarray, t: int,
                         learning_rate: float) -> np.ndarray:
    """
    Performs backward propagation using the Adam optimizer and L2 regularization.
    """
    # Calculate gradients for weights and biases
    grad_weights = np.dot(layer_input.T, output_grad)
    grad_biases = np.sum(output_grad, axis=0)

    # Add L2 regularization gradient (weight decay)
    grad_weights += L2_LAMBDA * layer.weights

    # Calculate input gradient for the previous layer
    input_grad = np.dot(output_grad, layer.weights.T)

    # Update parameters using Adam
    adam_optimizer_update(layer, grad_weights, grad_biases, t, learning_rate)

    return input_grad


def train(net: Network, inputs: np.ndarray, one_hot_labels: np.ndarray, t: int, learning_rate: float) -> float:
    """Trains the network for one batch and returns the loss."""
    # 1. Forward Pass
    hidden1_out = relu(forward_propagation(net.hidden1, inputs))
    hidden2_out = relu(forward_propagation(net.hidden2, hidden1_out))
    final_output = softmax(forward_propagation(net.output, hidden2_out))

    # 2. Calculate Loss (Cross-Entropy)
    # This calculation is combined with training to avoid a second forward pass
    batch_size = len(one_hot_labels)
    log_probs = -np.log(final_output[np.arange(batch_size), np.argmax(one_hot_labels, axis=1)])
    loss = np.sum(log_probs) / batch_size

    # Add L2 regularization loss
    l2_loss = (L2_LAMBDA / 2) * (
                np.sum(np.square(net.hidden1.weights)) + np.sum(np.square(net.hidden2.weights)) + np.sum(
            np.square(net.output.weights)))
    total_loss = loss + l2_loss

    # 3. Backward Pass (Backpropagation)
    output_grad = (final_output - one_hot_labels) / batch_size

    hidden2_grad = backward_propagation(net.output, hidden2_out, output_grad, t, learning_rate)
    hidden2_grad *= (hidden2_out > 0)  # ReLU derivative

    hidden1_grad = backward_propagation(net.hidden2, hidden1_out, hidden2_grad, t, learning_rate)
    hidden1_grad *= (hidden1_out > 0)  # ReLU derivative

    backward_propagation(net.hidden1, inputs, hidden1_grad, t, learning_rate)

    return total_loss


def predict(net: Network, inputs: np.ndarray) -> np.ndarray:
    """Makes predictions with the deeper network."""
    hidden1_out = relu(forward_propagation(net.hidden1, inputs))
    hidden2_out = relu(forward_propagation(net.hidden2, hidden1_out))
    final_output = softmax(forward_propagation(net.output, hidden2_out))
    return np.argmax(final_output, axis=1)


def main():
    # Load data
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")
    training_data = InputData(training_images, training_labels, num_images)

    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")
    test_data = InputData(test_images, test_labels, num_test_images)

    # Initialize the deeper network
    net = Network(
        hidden1=Layer.initialize(inputs=INPUT_SIZE, outputs=HIDDEN_SIZE_1),
        hidden2=Layer.initialize(inputs=HIDDEN_SIZE_1, outputs=HIDDEN_SIZE_2),
        output=Layer.initialize(inputs=HIDDEN_SIZE_2, outputs=OUTPUT_SIZE)
    )

    t = 0  # Timestep for Adam optimizer
    current_learning_rate = LEARNING_RATE

    for epoch in range(EPOCHS):
        # Shuffle training data each epoch to reduce variance
        permutation = np.random.permutation(num_images)
        training_data.images = training_data.images[permutation]
        training_data.labels = training_data.labels[permutation]

        total_loss = 0.0
        for i in range(0, num_images, BATCH_SIZE):
            t += 1  # Increment Adam timestep
            batch_images = training_data.images[i:i + BATCH_SIZE]
            batch_labels = training_data.labels[i:i + BATCH_SIZE]
            one_hot_labels = np.eye(OUTPUT_SIZE)[batch_labels]

            batch_loss = train(net, batch_images, one_hot_labels, t, current_learning_rate)
            total_loss += batch_loss * len(batch_images)

        # Apply learning rate decay
        current_learning_rate *= LR_DECAY

        # Evaluate accuracy on test data
        predictions = predict(net, test_data.images)
        accuracy = np.mean(predictions == test_data.labels)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Accuracy: {accuracy * 100:.2f}%, Avg Loss: {total_loss / num_images:.4f}, LR: {current_learning_rate:.6f}")


if __name__ == '__main__':
    main()