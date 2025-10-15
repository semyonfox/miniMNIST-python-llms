from dataclasses import dataclass
import numpy as np

# Neural Network Configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 60


@dataclass
class Layer:
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def initialize(cls, inputs: int, outputs: int):
        stddev = np.sqrt(2.0 / inputs)
        weights = np.random.normal(0.0, stddev, (inputs, outputs)).astype(np.float32)
        biases = np.zeros(outputs, dtype=np.float32)
        return cls(weights, biases)


@dataclass
class Network:
    hidden: Layer
    output: Layer


@dataclass
class InputData:
    images: np.ndarray
    labels: np.ndarray
    image_count: int


def load_mnist_images(filename: str) -> tuple[int, np.ndarray]:
    """Optimized MNIST image loading."""
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return num_images, (images.astype(np.float32) / 255.0)


def load_mnist_labels(filename: str) -> tuple[int, np.ndarray]:
    """Optimized MNIST label loading."""
    with open(filename, 'rb') as f:
        _, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return num_labels, labels


def forward_propagation(layer: Layer, inputs: np.ndarray) -> np.ndarray:
    """Optimized forward propagation using @ operator."""
    return inputs @ layer.weights + layer.biases


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """In-place ReLU for better memory efficiency."""
    np.maximum(x, 0, out=x)
    return x


def backward_propagation(layer: Layer, input: np.ndarray, output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
    """Optimized backward propagation."""
    batch_size = input.shape[0]

    # Optimized weight update
    layer.weights -= (input.T @ output_grad) * (learning_rate / batch_size)

    # Optimized bias update
    layer.biases -= np.sum(output_grad, axis=0) * (learning_rate / batch_size)

    # Return input gradient for next layer
    return output_grad @ layer.weights.T


def train(net: Network, input: np.ndarray, label: np.ndarray, learning_rate: float) -> float:
    """Optimized training function that returns batch loss."""
    batch_size = input.shape[0]

    # Forward pass
    hidden_preact = forward_propagation(net.hidden, input)
    hidden_output = relu(hidden_preact)

    output_preact = forward_propagation(net.output, hidden_output)
    final_output = softmax(output_preact)

    # Compute loss for monitoring
    correct_log_probs = -np.log(np.take_along_axis(final_output, label[:, None], axis=1) + 1e-10)
    batch_loss = np.sum(correct_log_probs)

    # Create one-hot encoded labels for gradient calculation
    one_hot_labels = np.eye(OUTPUT_SIZE, dtype=np.float32)[label]

    # Backward pass - fixed gradient calculation
    output_grad = final_output - one_hot_labels

    hidden_grad = backward_propagation(net.output, hidden_output, output_grad, learning_rate)
    hidden_grad *= (hidden_output > 0)  # ReLU derivative

    backward_propagation(net.hidden, input, hidden_grad, learning_rate)

    return batch_loss


def predict(net: Network, input: np.ndarray) -> np.ndarray:
    """Optimized prediction function."""
    hidden_output = relu(forward_propagation(net.hidden, input))
    final_output = softmax(forward_propagation(net.output, hidden_output))
    return np.argmax(final_output, axis=1)


def main():
    # Load data
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")

    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")

    # Initialize network
    net = Network(
        hidden=Layer.initialize(INPUT_SIZE, HIDDEN_SIZE),
        output=Layer.initialize(HIDDEN_SIZE, OUTPUT_SIZE)
    )

    # Training loop with progress tracking
    for epoch in range(EPOCHS):
        total_loss = 0.0

        # Shuffle training data each epoch
        indices = np.random.permutation(num_images)
        shuffled_images = training_images[indices]
        shuffled_labels = training_labels[indices]

        for i in range(0, num_images, BATCH_SIZE):
            batch_images = shuffled_images[i:i + BATCH_SIZE]
            batch_labels = shuffled_labels[i:i + BATCH_SIZE]

            batch_loss = train(net, batch_images, batch_labels, LEARNING_RATE)
            total_loss += batch_loss

        # Evaluate on test set
        test_predictions = predict(net, test_images)
        accuracy = np.mean(test_predictions == test_labels)
        avg_loss = total_loss / num_images

        print(f"Epoch {epoch + 1:2d}, Accuracy: {accuracy * 100:5.2f}%, Avg Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    main()