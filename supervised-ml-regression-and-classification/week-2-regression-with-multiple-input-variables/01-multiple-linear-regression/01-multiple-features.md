## **Multiple Linear Regression**

This model, also known as **linear regression with multiple features**, is an extension of the model from Week 1. It allows us to use more than one input feature to make more accurate predictions.

### **Motivation 🤔**

Instead of predicting a house's price based only on its size, we can use additional information to improve our model.

- **Single Feature (Week 1):** `x` = size
- **Multiple Features (Week 2):**
  - $x_1$ = size
  - $x_2$ = number of bedrooms
  - $x_3$ = number of floors
  - $x_4$ = age of home

---

### **New Notation ✍️**

With multiple features, we need to update our notation to handle them.

- **$x_j$**: Denotes the **j-th feature** in our feature list (e.g., $x_2$ is the number of bedrooms).
- **n**: The **total number of features** (in our example, n=4).
- **$\vec{x}^{(i)}$**: Represents the **i-th training example**. This is now a **vector**, or a list of numbers, containing all the feature values for that example.
  - For example, $\vec{x}^{(2)} = [1416, 3, 2, 40]$.
- **$x_j^{(i)}$**: The value of the **j-th feature** in the **i-th training example**.
  - For example, $x_3^{(2)} = 2$ (the number of floors in the second house).

---

### **The Updated Model Equation**

The model is extended from a simple line to an equation that incorporates all `n` features.

- **Model with `n` features:**
  $$f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

To write this more compactly, we use vector notation and the **dot product**.

1. **Define Parameter and Feature Vectors:**

   - The parameters `w` are now a vector: $\vec{w} = [w_1, w_2, \dots, w_n]$
   - The features `x` for a single example are a vector: $\vec{x} = [x_1, x_2, \dots, x_n]$
   - `b` remains a single number (a scalar).

2. **Use the Dot Product:** The dot product of $\vec{w}$ and $\vec{x}$ is defined as:
   $$\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + \dots + w_nx_n$$

3. **The Final, Compact Model:**
   $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
