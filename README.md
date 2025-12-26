# Convolutional Neural Network (CNN) Project

## About the Project

This project focuses on building a **Convolutional Neural Network (CNN)** to classify images from the **Fashion MNIST dataset**. The main goal is to train an agent to correctly identify fashion items and maximize classification accuracy by experimenting with different CNN architectures and hyperparameters.

The Fashion MNIST dataset consists of:

- **Training set:** 60,000 images
- **Testing set:** 10,000 images
- **Number of classes:** 10 (labels 0–9, each corresponding to a fashion category)
- **Image size:** 28x28 pixels
- **Pixel values:** 0–255, rescaled to 0–1 for training

---

## CNN Architecture

The model architecture includes:

1. **Conv2D Layer:** Extracts features from images
2. **MaxPooling Layer:** Reduces spatial dimensions
3. **Dense Layer:** Fully connected layer to combine features
4. **Output Layer:** Classifies into 10 categories
5. **Dropout Layers:** Prevent overfitting

We experimented with multiple architectures, including additional Conv2D and Dense layers, but the **best-performing model** used a single Conv2D layer with 3x3 kernel size, L2 regularization, and dropout layers. This achieved a **testing accuracy of over 92%**.

**Hyperparameters used in the best model:**

- Optimizer: Adam
- Activation: ReLU
- Initializer: HeNormal
- Dropout: 0.5
- Kernel size: 3x3
- Padding: Same

---

## Hyperparameter Experiments

### 1. Dropout

| Dropout | Accuracy (%) |
| ------- | ------------ |
| 0.2     | 91.6         |
| 0.4     | 91.1         |
| 0.5     | 92.1         |
| 0.6     | 91.01        |

### 2. Optimizers

| Optimizer | Accuracy (%) |
| --------- | ------------ |
| SGD       | 88.35        |
| RMSprop   | 87.5         |
| Adadelta  | 71.66        |

### 3. Activation Functions

| Activation | Accuracy (%) |
| ---------- | ------------ |
| SELU       | 90.7         |
| GELU       | 91.32        |
| Tanh       | 90.38        |

### 4. Initializers

| Initializer   | Accuracy (%) |
| ------------- | ------------ |
| GlorotNormal  | 91.2         |
| GlorotUniform | 91.14        |
| RandomNormal  | 91.54        |

### 5. Kernel Sizes

| Kernel Size | Accuracy (%) |
| ----------- | ------------ |
| 1x1         | 86.16        |
| 2x2         | 91.42        |
| 4x4         | 91.57        |

### 6. Padding

| Conv Padding | MaxPooling Padding | Accuracy (%) |
| ------------ | ------------------ | ------------ |
| same         | same               | 91.39        |
| same         | valid              | 91.43        |
| valid        | valid              | 91.28        |

---

## Observations & Improvements

- Using **dropouts** significantly reduced overfitting and improved generalization.
- **L2 regularization** helped in stabilizing training.
- Changing the **kernel size** and **padding** slightly affected accuracy.
- Adam optimizer consistently performed better than SGD, RMSprop, and Adadelta.
- Activation functions like GELU slightly outperformed SELU and Tanh in this setup.

---

## Results

- **Highest Testing Accuracy:** 92.1%
- Best configuration: Conv2D (3x3, padding=same), Dropout=0.5, Optimizer=Adam, Activation=ReLU, Initializer=HeNormal
