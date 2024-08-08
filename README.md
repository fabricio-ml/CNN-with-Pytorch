# CNN with Pytorch

This PyTorch script implements a CNN for MNIST digit classification. It defines a model with two convolutional layers and two fully connected layers, loads and preprocesses MNIST data, trains the network for 5 epochs using Adam optimizer and Cross Entropy Loss, and evaluates accuracy on the test set. The code demonstrates the complete workflow of training and testing a neural network for image classification.

## Prerequisites

1. Python (3.x recommended)
2. PyTorch
3. torchvision

## Additionally, you'll need:

CUDA-capable GPU (optional, for faster training)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fabricio-ml/CNN-with-Pytorch.git
   cd CNN-with-Pytorch

2. Install the dependencies:
   ```bash
   pip install torch torchvision
   
2. Run it!
   ```bash
   python convo_net.py
