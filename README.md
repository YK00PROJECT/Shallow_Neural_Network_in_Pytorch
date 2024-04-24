# MNIST Digit Classification using PyTorch

## Overview
This project involves the development of a neural network using PyTorch to classify handwritten digits from the MNIST dataset. It demonstrates the entire process from loading and pre-processing data to defining and training a neural network model.

## Repository Contents
- **train_loader**: DataLoader for training data
- **test_loader**: DataLoader for testing data
- **model**: A simple neural network model with one hidden layer
- **training script**: Script to train the neural network

## Installation
To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/mnist-digit-classification.git
   cd mnist-digit-classification
   ```

2. **Install Required Libraries**:
   Make sure you have Python and pip installed, then run:
   ```
   pip install torch torchvision matplotlib
   ```

## Usage
- **Load and Pre-process Data**: The MNIST dataset is automatically downloaded using PyTorch's `torchvision.datasets` module. It is transformed to tensors using `transforms.ToTensor()`, and data loaders are created for both training and test sets.
- **Model Training**: Run the training script to train the model. Adjust hyperparameters like learning rate and number of epochs in the script as needed.
- **Evaluation**: After training, evaluate the model on the test dataset to see how well it performs on unseen data.

## Model Architecture
The model consists of the following layers:
- **Input Layer**: Accepts flattened 28x28 pixel MNIST images, resulting in 784 input features.
- **Hidden Layer**: A dense (fully connected) layer with 64 neurons and a Sigmoid activation function.
- **Output Layer**: A dense layer with 10 neurons (one for each digit) that outputs the classification scores for each class.

## Training
- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.01.
- **Loss Function**: Cross Entropy Loss, which combines LogSoftmax and NLLLoss in one single class.
- **Metrics**: Accuracy is used to evaluate the model performance during training.



## Contributing
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

This README provides all the essential information needed for anyone interested in using or contributing to the project. It includes installation instructions, a detailed usage guide, a brief overview of the model architecture, and how to train the model.
