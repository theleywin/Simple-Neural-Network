# Neural Network for Handwritten Digit Recognition

This project implements a simple neural network from scratch using Python and NumPy to classify handwritten digits from the MNIST dataset.

## Table of Contents
- [Neural Network for Handwritten Digit Recognition](#neural-network-for-handwritten-digit-recognition)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Dataset](#dataset)
  - [Neural Network Architecture](#neural-network-architecture)
    - [Activation Functions](#activation-functions)
  - [Implementation Details](#implementation-details)
    - [Preprocessing](#preprocessing)
    - [Functions](#functions)
  - [Usage](#usage)

## Project Description
This project demonstrates how to build, train, and evaluate a neural network to recognize handwritten digits using the MNIST dataset. The network is implemented without external machine learning libraries to highlight fundamental concepts.

## Dataset
The dataset used in this project is a subset of the MNIST dataset. It contains grayscale images of handwritten digits (0-9). Each image is represented as a 28x28 pixel grid, flattened into a 784-dimensional vector.

- Training data: `dataset/train-1.csv`
- Validation data: Split from the training data during preprocessing.

## Neural Network Architecture
The neural network has the following structure:
1. **Input Layer**: 784 units (one for each pixel of the image).
2. **Hidden Layer**: 10 units with ReLU activation.
3. **Output Layer**: 10 units (one for each digit class) with softmax activation.

### Activation Functions
- **ReLU**: Rectified Linear Unit for non-linearity in the hidden layer.
- **Softmax**: Converts output scores into probabilities for classification.

## Implementation Details
### Preprocessing
- The input data is normalized by dividing pixel values by 255 to scale them to the range [0, 1].
- The dataset is split into training and validation sets.

### Functions
- `init_params`: Initializes weights and biases randomly.
- `ReLU` and `ReLU_deriv`: Implements the ReLU activation and its derivative.
- `softmax`: Computes the softmax probabilities.
- `forward_prop`: Performs forward propagation through the network.
- `backward_prop`: Computes gradients using backpropagation.
- `update_params`: Updates weights and biases using gradient descent.
- `gradient_descent`: Optimizes the model parameters over multiple iterations.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/theleywin/Simple-Neural-Network.git
   cd Simple-Neural-Network
