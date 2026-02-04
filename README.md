# Rust MNIST Neural Network

A simple neural network implementation in Rust for handwritten digit recognition using the MNIST dataset.

## Overview

This project implements a basic feedforward neural network from scratch using Rust and the `ndarray` crate. The network learns to recognize handwritten digits (0-9) from the MNIST dataset with approximately 80% accuracy.

## Features

- **Simple Neural Network Architecture**: 784 → 128 → 10 neurons
- **ReLU Activation**: Fast and effective non-linear activation function
- **Backpropagation**: Standard gradient descent with learning rate decay
- **MNIST Dataset**: Automatic download and loading of the famous handwritten digit dataset
- **Visual Output**: ASCII art visualization of digits with predictions

## Requirements

- Rust (latest stable version)
- Cargo (comes with Rust)

## Dependencies

```toml
[dependencies]
mnist_reader = "0.1.1"
ndarray = "0.17.2"
ndarray-rand = "0.16.0"
```

## Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd rust-MNIST
   ```

## Usage

Run the project in release mode for better performance:

```bash
cargo run --release
```

Or in debug mode:

```bash
cargo run
```

## How It Works

### Neural Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9)

### Training Process

1. **Data Preparation**: 
   - Loads MNIST training data (60,000 images)
   - Normalizes pixel values to [0, 1] range
   - Converts labels to one-hot encoded vectors

2. **Training**:
   - Uses first 5,000 samples
   - Trains for 10 epochs
   - Learning rate: 0.05 with decay (lr / (1 + epoch × 0.1))
   - Backpropagation with gradient descent

3. **Testing**:
   - Evaluates on 1,000 test samples
   - Displays first 5 predictions with ASCII visualizations
   - Reports overall test accuracy

## Sample Output

```
Loading MNIST dataset...
Train data size: 60000
Test data size: 10000

Creating neural network (784 -> 128 -> 10)...
Training neural network...
Epoch 2 (lr=0.0455): Training accuracy = 72.52%
Epoch 4 (lr=0.0385): Training accuracy = 77.74%
Epoch 6 (lr=0.0333): Training accuracy = 80.02%
Epoch 8 (lr=0.0294): Training accuracy = 81.24%
Epoch 10 (lr=0.0263): Training accuracy = 82.34%
Training complete!

=== First 5 Training Samples ===

--- Sample 0 ---
Predicted: 5  |  Actual: 5  |  ✓ Correct
[ASCII art of digit 5]

Test Accuracy: 794/1000 (79.40%)
```

## Performance

- **Training Accuracy**: ~82% (on 5,000 samples)
- **Test Accuracy**: ~79-80% (on 1,000 samples)
- **Training Time**: ~10-15 seconds in release mode

## Project Structure

```
rust-MNIST/
├── Cargo.toml          # Project dependencies
├── README.md           # This file
├── src/
│   └── main.rs         # Neural network implementation
├── mnist-data/         # MNIST dataset (auto-downloaded)
└── target/             # Build artifacts
```

## Implementation Details

### Key Components

1. **NeuralNetwork Struct**: Holds network parameters (weights)
2. **feedforward()**: Forward pass through the network
3. **backpropagation()**: Backward pass for weight updates
4. **predict()**: Returns predicted digit class

### Activation Function

ReLU (Rectified Linear Unit): `f(x) = max(0, x)`
- Simple and computationally efficient
- Helps prevent vanishing gradient problem

### Weight Initialization

Weights are randomly initialized using uniform distribution in range [-1.0, 1.0].

## Possible Improvements

- [ ] Add softmax activation for output layer
- [ ] Implement cross-entropy loss
- [ ] Add momentum or Adam optimizer
- [ ] Support for multiple hidden layers
- [ ] Batch training instead of single-sample updates
- [ ] Save/load trained model
- [ ] Data augmentation
- [ ] Dropout for regularization
- [ ] Learning rate scheduling
- [ ] Validation set for hyperparameter tuning

## License

MIT

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- `mnist_reader` crate for easy dataset access
- `ndarray` crate for numerical operations
