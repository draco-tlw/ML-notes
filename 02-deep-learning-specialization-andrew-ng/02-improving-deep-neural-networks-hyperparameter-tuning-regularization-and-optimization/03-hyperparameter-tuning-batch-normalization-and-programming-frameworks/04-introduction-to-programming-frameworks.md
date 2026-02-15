# Deep Learning Frameworks

### Motivation for Using Frameworks

- **Transition from Scratch to Frameworks:** While implementing algorithms from scratch (e.g., using Python and NumPy) is crucial for understanding the underlying mechanics, it becomes impractical for complex applications.
- **Scalability and Complexity:** As models grow in size or complexity (e.g., **Convolutional Neural Networks (CNNs)** or **Recurrent Neural Networks (RNNs)**), manual implementation is inefficient.
- **Abstraction Analogy:** using a deep learning framework is analogous to using a linear algebra library for matrix multiplication rather than coding the multiplication function yourself; it is about utilizing optimized tools for efficiency.

### Criteria for Choosing a Framework

When selecting a deep learning framework, consider the following three key criteria:

1. **Ease of Programming:**

- Includes the ease of developing and iterating on the neural network.
- Includes the ease of **deploying** to production for actual use (scaling to thousands or millions of users).

2. **Running Speeds:**

- Focus on the efficiency of **training** on large datasets.
- Performance varies between frameworks regarding how fast they can train specific architectures.

3. **True Openness:**

- **Open Source vs. Open Governance:** A framework should be not only open source but also have good governance.
- **Vendor Lock-in Risk:** Be cautious of frameworks controlled by a single corporation, as they may gradually close off features or migrate functionality to proprietary cloud services.
- **Trust:** Consider how much you trust that the framework will remain truly open for the long term.

### Summary

- Frameworks provide a **higher level of abstraction** than numerical linear algebra libraries.
- There are many credible frameworks available (e.g., supporting Python, Java, C++), and the "best" choice often depends on the specific application (Computer Vision, NLP, etc.) and language preference.
- Adopting a framework significantly improves efficiency when building machine learning applications.

---

# Introduction to TensorFlow

### Overview

**TensorFlow** is a powerful deep learning framework that allows developers to implement complex neural networks efficiently. Its primary advantage is **automatic differentiation**, meaning you only need to implement the **forward propagation** (the cost function), and TensorFlow automatically calculates the gradients for **backpropagation**.

### 1. Basic Program Structure

To minimize a cost function (e.g., $J(w) = w^2 - 10w + 25$), a typical TensorFlow program follows these steps:

1.  **Initialization:** Import libraries and define trainable parameters.
    - `import tensorflow as tf`
    - **tf.Variable:** Used to define parameters that the optimizer will modify (e.g., weights $w$).
2.  **Optimizer Selection:** Define the optimization algorithm.
    - **Adam Optimizer** is commonly used (`tf.keras.optimizers.Adam`).
3.  **Cost Function Definition:** Write the mathematical expression for the cost $J$.
4.  **Training Step:** \* **tf.GradientTape:** Records the sequence of operations in the forward pass to later compute gradients in reverse.
    - **Optimizer.apply_gradients:** Updates the variables based on the calculated gradients.

### 2. Handling Training Data

In real-world scenarios, the cost function depends on both parameters ($w, b$) and data ($x, y$).

- Data can be passed into the cost function as arrays (e.g., NumPy arrays).
- By making the coefficients of the cost function dependent on an input array `x`, you can change the objective function dynamically without rewriting the optimization logic.

### 3. Key Syntax & API Examples

| Component        | TensorFlow Implementation (v2.x)             |
| :--------------- | :------------------------------------------- |
| **Variable**     | `w = tf.Variable(0, dtype=tf.float32)`       |
| **Optimizer**    | `optimizer = tf.keras.optimizers.Adam(0.1)`  |
| **Forward Pass** | `with tf.GradientTape() as tape: cost = ...` |
| **Gradients**    | `grads = tape.gradient(cost, [w])`           |
| **Update**       | `optimizer.apply_gradients(zip(grads, [w]))` |

### 4. The Computation Graph

TensorFlow operates by constructing a **Computation Graph**:

- **Nodes:** Represent mathematical operations (addition, multiplication, squaring).
- **Edges:** Represent the data (Tensors) flowing between operations.
- **Automatic Backprop:** Because TensorFlow has a map of the forward operations (the graph), it can automatically traverse the graph backward to calculate derivatives for any parameter.

> **Note:** One of the greatest strengths of frameworks like TensorFlow is **modularity**. For example, you can switch from an Adam optimizer to a Gradient Descent optimizer by changing only a single line of code.

---
