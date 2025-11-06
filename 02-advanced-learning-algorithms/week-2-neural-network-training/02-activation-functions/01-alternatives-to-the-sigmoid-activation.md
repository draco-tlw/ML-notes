### **Beyond the Sigmoid Function**

So far, we have only used the **sigmoid activation function** ($g(z) = \frac{1}{1 + e^{-z}}$). This was a natural starting point because it's the foundation of logistic regression.

However, the sigmoid function is not always the best choice. Its output is always between 0 and 1, which is useful for probabilities (like in an output layer) but can be too restrictive for hidden layers.

- **Motivation:** Consider a hidden neuron trying to learn the "awareness" of a T-shirt. Awareness isn't necessarily a binary (0/1) value or a probability. It could be a number that grows (e.g., from 0 to a very large value representing "gone viral"). A sigmoid function cannot represent this.

---

### **The ReLU Activation Function**

To model values that can be zero or any positive number, a different activation function is more effective: **ReLU (Rectified Linear Unit)**.

- **Definition:** The ReLU function is defined as **`g(z) = max(0, z)`**.
- **How it Works:**
  - If the input `z` is negative, the output is `0`.
  - If the input `z` is positive, the output is just `z`.
- **The Graph:** It's a two-part function: a flat line at 0 for all negative inputs, and a 45-degree line for all positive inputs.
- **Output Range:** The output of a ReLU neuron is always 0 or a positive number (it is **non-negative**). This makes it an excellent default choice for hidden layers.

---

### **Commonly Used Activation Functions**

Here are the three most common activation functions you will encounter:

1.  **Sigmoid:**
    - **Formula:** $g(z) = \frac{1}{1 + e^{-z}}$
    - **Output Range:** 0 to 1.
    - **Use Case:** Good for the **output layer** in a **binary classification** problem (where you need a probability).

2.  **ReLU (Rectified Linear Unit):**
    - **Formula:** $g(z) = \max(0, z)$
    - **Output Range:** 0 to $\infty$ (non-negative).
    - **Use Case:** The most common and effective default choice for **hidden layers**.

3.  **Linear:**
    - **Formula:** $g(z) = z$
    - **Output Range:** $-\infty$ to $+\infty$ (any number).
    - **Note:** This is often referred to as "no activation function" since the output `a` is simply equal to the input `z`.
    - **Use Case:** Good for the **output layer** in a **regression** problem (where you need to predict a number, not a probability).

_(A fourth function, **softmax**, will be introduced later for multi-class classification.)_
