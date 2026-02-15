# Note: Deep Neural Networks

**Source:** Coursera - Neural Networks and Deep Learning (Week 4)

## 1. Deep vs. Shallow Networks

- **Logistic Regression:** Considered a "shallow" model (1 layer).

- **Shallow Neural Network:** A network with a single hidden layer (2 layers).

- **Deep Neural Network:** A network with multiple hidden layers.

- **Significance:** Deep networks can learn functions that shallow models are often unable to learn effectively.

## 2. Notation for Deep Networks

We utilize specific notation to handle multiple layers (L).

- **L:** The total number of layers in the network.

- **n^{[l]}:** The number of nodes (units) in layer l.

- n^{[0]} = n_x (Input layer size).

- n^{[L]} (Output layer size).

- **a^{[l]}:** The activations in layer l.

- a^{[0]} = x (Input features).

- a^{[L]} = \hat{y} (Predicted output).

- **W^{[l]}, b^{[l]}:** Weights and bias parameters for computing z^{[l]} in layer l.

## 3. Forward Propagation

Forward propagation calculates the output predictions. Unlike other steps, using an explicit `for` loop (from layer 1 to L) is necessary and acceptable here.

### General Equations (Layer l)

- **Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}**.

- **A^{[l]} = g^{[l]}(Z^{[l]})** where g^{[l]} is the activation function for layer l.

- **Initialization:** A^{[0]} = X (Training examples stacked in columns).

## 4. Debugging: Matrix Dimensions

Checking matrix dimensions is a primary tool for debugging deep networks.

### Parameters (W and b)

- **W^{[l]} Dimension:** (n^{[l]}, n^{[l-1]}).

- **b^{[l]} Dimension:** (n^{[l]}, 1).

- _Rule:_ To calculate layer l, you multiply by the previous layer size (n^{[l-1]}) to get the current layer size (n^{[l]}).

- **Derivatives (dW, db):** Always have the same dimensions as W and b respectively.

### Activations (Z and A)

- **Vectorized Dimension:** (n^{[l]}, m), where m is the number of training examples.

- **Broadcasting:** b^{[l]} is broadcasted from (n^{[l]}, 1) to (n^{[l]}, m) during calculation.

## 5. Why Deep Representations?

Why do deep networks often outperform shallow ones?

### A. Hierarchical Feature Learning

- **Simple to Complex:** Earlier layers detect simple functions (e.g., edges in an image, waveform features in audio).

- **Composition:** Later layers compose these simple features to detect complex objects (e.g., eyes/noses \rightarrow faces; phonemes \rightarrow words \rightarrow sentences).

### B. Circuit Theory

- **Efficiency:** There are mathematical functions that are computationally easier to calculate with deep networks than shallow ones.

- **Example (Parity/XOR):** Computing the parity of N bits takes O(\log N) layers in a deep network (XOR tree).

- **Shallow Cost:** To compute the same function with a single hidden layer requires exponentially many (2^{N-1}) hidden units.

## 6. Implementation: Building Blocks

Implementing a deep network involves modular "Forward" and "Backward" functions for each layer.

### The Cache

- **Purpose:** During forward propagation, we compute values (like Z^{[l]}) that are required later for backward propagation.

- **Mechanism:** We store ("cache") these values during the forward pass to pass them into the backward functions.

### Backward Propagation Formulas

The backpropagation steps calculate gradients. Note that \* denotes element-wise multiplication.

1. **Output Layer (L) Initialization:**

- dZ^{[L]} = A^{[L]} - Y.

2. **Gradients for Layer l:**

- dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}.

- db^{[l]} = \frac{1}{m} \text{np.sum}(dZ^{[l]}, axis=1, keepdims=True).

- dZ^{[l-1]} = W^{[l]T} dZ^{[l]} \* g'^{[l-1]}(Z^{[l-1]}).

- _Note:_ A^{[0]T} is equivalent to X^T.

## 7. Hyperparameters vs. Parameters

Applying deep learning is an empirical process requiring experimentation with hyperparameters.

| **Parameters** (Learned by the model) | **Hyperparameters** (Set by the human) |
| ------------------------------------- | -------------------------------------- |
| W (Weights)                           |                                        |
| b (Biases)                            |                                        |
|                                       | Number of iterations                   |
|                                       | \alpha (Learning Rate)                 |
|                                       | L (Number of hidden layers)            |
|                                       | n^{[l]} (Hidden units per layer)       |
|                                       | Choice of activation functions         |

- **Guidance:** You cannot know the best hyperparameters in advance; you must iterate, test, and evaluate.

## 8. Deep Learning and the Brain

- **Analogy:** There is a loose analogy between a logistic regression unit and a biological neuron (receiving signals, thresholding, firing).

- **Limitations:** Neuroscientists do not fully understand how single neurons work or learn. The brain analogy is simplified and less useful for modern deep learning engineering.
