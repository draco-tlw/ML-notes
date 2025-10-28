## **Implementing Neural Networks in TensorFlow**

**TensorFlow** is one of the most popular open-source frameworks used to build and train deep learning models. This course will focus on TensorFlow for its implementations (an alternative framework is PyTorch).

We will use TensorFlow to implement **inference** (the process of making a prediction) by defining the layers of our network.

### **The `Dense` Layer**

In TensorFlow, the standard, fully-connected layer we have been studying is called a **`Dense`** layer.

- **Definition:** A `Dense` layer means that every neuron in that layer is connected to _every_ activation from the previous layer.
- **Syntax:** You define a layer using `Dense(units=..., activation=...)`
  - **`units`**: The number of neurons in the layer.
  - **`activation`**: The activation function to be used by the neurons in that layer (e.g., `'sigmoid'`).

---

### **Forward Propagation in TensorFlow**

The process of feeding an input `x` through the layers to compute the final output is called **forward propagation**.

#### **Example 1: Coffee Roasting (2-layer network)**

- **Goal:** Predict if coffee is good (1) or bad (0) based on 2 features.
- **Input (Layer 0):** $\vec{x} = [200, 17]$ (temperature, duration)

<!-- end list -->

```python
# Layer 1 (Hidden Layer)
# Define the layer's structure (3 neurons, sigmoid)
Layer1 = Dense(units=3, activation='sigmoid')
# Compute the layer's output (forward prop)
a1 = Layer1(x)  # a1 will be a vector with 3 activations

# Layer 2 (Output Layer)
# Define the layer's structure (1 neuron, sigmoid)
Layer2 = Dense(units=1, activation='sigmoid')
# Compute the final output
a2 = Layer2(a1) # a2 is the final probability (e.g., 0.8)

# Optional: Make a binary prediction
if a2 >= 0.5:
  y_hat = 1
else:
  y_hat = 0
```

#### **Example 2: Handwritten Digits (3-layer network)**

This example shows how to stack multiple layers.

- **Input (Layer 0):** $\vec{x}$ (a vector of pixel intensity values)

<!-- end list -->

```python
# Layer 1 (Hidden)
Layer1 = Dense(units=25, activation='sigmoid')
a1 = Layer1(x)

# Layer 2 (Hidden)
Layer2 = Dense(units=15, activation='sigmoid')
a2 = Layer2(a1)

# Layer 3 (Output)
Layer3 = Dense(units=1, activation='sigmoid')
a3 = Layer3(a2)  # a3 is the final prediction
```

**Note:** The lab exercises will provide the code for loading the TensorFlow library and the pre-trained parameters (`w` and `b`) for each layer. The code above defines the _architecture_ and executes the _forward propagation_ steps.
