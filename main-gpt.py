from dataclasses import dataclass
import numpy as np

# Neural Network Configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 60
L2_LAMBDA = 1e-4  # L2 regularization strength

@dataclass
class Layer:
    weights: np.ndarray
    biases: np.ndarray
    m_w: np.ndarray = None
    v_w: np.ndarray = None
    m_b: np.ndarray = None
    v_b: np.ndarray = None

    @classmethod
    def initialize(cls, inputs: int, outputs: int, is_output=False):
        """Initialize weights and biases with He or Xavier depending on layer type"""
        if is_output:
            stddev = np.sqrt(1.0 / inputs)  # Xavier for output
        else:
            stddev = np.sqrt(2.0 / inputs)  # He for hidden
        weights = np.random.normal(0.0, stddev, size=(inputs, outputs))
        biases = np.zeros(outputs)
        # Initialize Adam optimizer variables
        m_w = np.zeros_like(weights)
        v_w = np.zeros_like(weights)
        m_b = np.zeros_like(biases)
        v_b = np.zeros_like(biases)
        return cls(weights, biases, m_w, v_w, m_b, v_b)

@dataclass
class Network:
    hidden: Layer
    output: Layer

@dataclass
class InputData:
    images: np.ndarray
    labels: np.ndarray
    image_count: int

# ------------------ Utility Functions ------------------ #

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

# ------------------ Activations ------------------ #

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def leaky_relu(x: np.ndarray, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x: np.ndarray, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)

# ------------------ Forward/Backward ------------------ #

def forward(layer: Layer, inputs: np.ndarray) -> np.ndarray:
    return np.dot(inputs, layer.weights) + layer.biases

def backward_adam(layer: Layer, inputs: np.ndarray, output_grad: np.ndarray, lr: float,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, t=1, l2_lambda=0.0):
    grad_w = np.dot(inputs.T, output_grad) / inputs.shape[0] + l2_lambda * layer.weights
    grad_b = np.mean(output_grad, axis=0)

    # Adam updates
    layer.m_w = beta1 * layer.m_w + (1 - beta1) * grad_w
    layer.v_w = beta2 * layer.v_w + (1 - beta2) * (grad_w ** 2)
    m_w_hat = layer.m_w / (1 - beta1 ** t)
    v_w_hat = layer.v_w / (1 - beta2 ** t)
    layer.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

    layer.m_b = beta1 * layer.m_b + (1 - beta1) * grad_b
    layer.v_b = beta2 * layer.v_b + (1 - beta2) * (grad_b ** 2)
    m_b_hat = layer.m_b / (1 - beta1 ** t)
    v_b_hat = layer.v_b / (1 - beta2 ** t)
    layer.biases -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    input_grad = np.dot(output_grad, layer.weights.T)
    return input_grad

# ------------------ Training / Prediction ------------------ #

def train(net: Network, inputs: np.ndarray, labels: np.ndarray, lr: float, t: int):
    # Forward
    hidden_linear = forward(net.hidden, inputs)
    hidden_output = leaky_relu(hidden_linear)
    final_linear = forward(net.output, hidden_output)
    final_output = softmax(final_linear)

    # Cross-entropy gradient
    output_grad = (final_output - labels) / inputs.shape[0]

    # Backward
    hidden_grad = backward_adam(net.output, hidden_output, output_grad, lr, t=t, l2_lambda=L2_LAMBDA)
    hidden_grad *= leaky_relu_grad(hidden_linear)
    backward_adam(net.hidden, inputs, hidden_grad, lr, t=t, l2_lambda=L2_LAMBDA)

def predict(net: Network, inputs: np.ndarray) -> np.ndarray:
    hidden_output = leaky_relu(forward(net.hidden, inputs))
    final_output = softmax(forward(net.output, hidden_output))
    return np.argmax(final_output, axis=1)

# ------------------ Main Loop ------------------ #

def main():
    # Load data
    num_images, training_images = load_mnist_images("train-images.idx3-ubyte")
    _, training_labels = load_mnist_labels("train-labels.idx1-ubyte")
    training_data = InputData(training_images, training_labels, num_images)

    num_test_images, test_images = load_mnist_images("t10k-images.idx3-ubyte")
    _, test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")
    test_data = InputData(test_images, test_labels, num_test_images)

    # Network initialization
    net = Network(
        hidden=Layer.initialize(INPUT_SIZE, HIDDEN_SIZE),
        output=Layer.initialize(HIDDEN_SIZE, OUTPUT_SIZE, is_output=True)
    )

    for epoch in range(1, EPOCHS + 1):
        # Shuffle training data
        indices = np.arange(num_images)
        np.random.shuffle(indices)
        training_data.images = training_data.images[indices]
        training_data.labels = training_data.labels[indices]

        total_loss = 0.0
        for i in range(0, num_images, BATCH_SIZE):
            batch_images = training_data.images[i:i+BATCH_SIZE]
            batch_labels = training_data.labels[i:i+BATCH_SIZE]
            one_hot_labels = np.eye(OUTPUT_SIZE)[batch_labels]

            # Train step
            t = epoch  # timestep for Adam
            train(net, batch_images, one_hot_labels, LEARNING_RATE, t=t)

            # Compute loss
            hidden_out = leaky_relu(forward(net.hidden, batch_images))
            out = softmax(forward(net.output, hidden_out))
            log_probs = np.log(np.clip(out[np.arange(len(batch_labels)), batch_labels], 1e-10, 1.0))
            total_loss -= np.sum(log_probs)

        # Evaluate
        predictions = predict(net, test_data.images)
        accuracy = np.mean(predictions == test_data.labels)
        print(f"Epoch {epoch}, Accuracy: {accuracy*100:.2f}%, Avg Loss: {total_loss/num_images:.4f}")

if __name__ == "__main__":
    main()
