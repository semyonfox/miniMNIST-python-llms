from dataclasses import dataclass
import numpy as np

# Neural Network Configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 512     # increased for more representational power
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 64
DECAY_RATE = 0.99     # learning rate decay per epoch

np.random.seed(42)    # reproducibility and consistent results

@dataclass
class Layer:
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def initialize(cls, inputs: int, outputs: int):
        # Kaiming He initialization (better stability for ReLU)
        stddev = np.sqrt(2.0 / inputs)
        weights = np.random.randn(inputs, outputs) * stddev
        biases = np.zeros(outputs)
        return cls(weights, biases)

@dataclass
class Network:
    hidden: Layer
    output: Layer

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(layer: Layer, inputs: np.ndarray) -> np.ndarray:
    return np.dot(inputs, layer.weights) + layer.biases

def backward(layer: Layer, input: np.ndarray, grad_output: np.ndarray, lr: float) -> np.ndarray:
    grad_weights = np.dot(input.T, grad_output) / input.shape[0]
    grad_biases = np.mean(grad_output, axis=0)
    input_grad = np.dot(grad_output, layer.weights.T)

    layer.weights -= lr * grad_weights
    layer.biases -= lr * grad_biases
    return input_grad

def batch_norm(x: np.ndarray, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm

def cross_entropy_loss(pred: np.ndarray, y_true: np.ndarray) -> float:
    clipped = np.clip(pred, 1e-10, 1.0)
    return -np.mean(np.log(clipped[np.arange(len(y_true)), y_true]))

def train_batch(net: Network, x: np.ndarray, y: np.ndarray, lr: float):
    # forward pass
    hidden_linear = forward(net.hidden, x)
    hidden_activated = relu(batch_norm(hidden_linear))
    output_linear = forward(net.output, hidden_activated)
    predictions = softmax(output_linear)

    # compute gradient of loss w.r.t. output
    one_hot = np.eye(OUTPUT_SIZE)[y]
    grad_output = (predictions - one_hot) / y.shape[0]

    # backward pass
    grad_hidden = backward(net.output, hidden_activated, grad_output, lr)
    grad_hidden *= relu_derivative(hidden_linear)
    backward(net.hidden, x, grad_hidden, lr)

    # compute loss
    return cross_entropy_loss(predictions, y)

def predict(net: Network, x: np.ndarray) -> np.ndarray:
    hidden = relu(forward(net.hidden, x))
    out = softmax(forward(net.output, hidden))
    return np.argmax(out, axis=1)

def main():
    from pathlib import Path

    def load_images(file: str):
        with open(file, 'rb') as f:
            _, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            return data.astype(np.float32) / 255.0

    def load_labels(file: str):
        with open(file, 'rb') as f:
            _, num = np.frombuffer(f.read(8), dtype='>i4')
            return np.frombuffer(f.read(), dtype=np.uint8)

    # Load data
    train_images = load_images("train-images.idx3-ubyte")
    train_labels = load_labels("train-labels.idx1-ubyte")
    test_images = load_images("t10k-images.idx3-ubyte")
    test_labels = load_labels("t10k-labels.idx1-ubyte")

    net = Network(
        hidden=Layer.initialize(INPUT_SIZE, HIDDEN_SIZE),
        output=Layer.initialize(HIDDEN_SIZE, OUTPUT_SIZE),
    )

    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(train_images))
        train_images, train_labels = train_images[idx], train_labels[idx]
        lr = LEARNING_RATE * (DECAY_RATE ** epoch)

        losses = []
        for i in range(0, len(train_images), BATCH_SIZE):
            x_batch = train_images[i:i + BATCH_SIZE]
            y_batch = train_labels[i:i + BATCH_SIZE]
            loss = train_batch(net, x_batch, y_batch, lr)
            losses.append(loss)

        preds = predict(net, test_images)
        acc = np.mean(preds == test_labels)
        print(f"Epoch {epoch+1} | Acc: {acc*100:.2f}% | Loss: {np.mean(losses):.4f}")

if __name__ == "__main__":
    main()
