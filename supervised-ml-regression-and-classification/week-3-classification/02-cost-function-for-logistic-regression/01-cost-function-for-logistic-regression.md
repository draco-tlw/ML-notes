## **The Cost Function for Logistic Regression**

The goal of a **cost function**, $J(\vec{w},b)$, is to measure how well a set of parameters $(\vec{w}, b)$ performs on the training data. A lower cost means a better fit.

---

### **Why Not Use the Squared Error Cost?**

1. **The Problem of Non-Convexity:**

   - If we use the same squared error cost function from linear regression, $J(\vec{w},b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$, with the logistic regression model's sigmoid function, the resulting cost function shape is **non-convex**.
   - A non-convex function has many "dips" or **local minima**.
   - This is a problem because **gradient descent** is not guaranteed to find the best possible parameters (the **global minimum**) and can get stuck in a suboptimal local minimum.

2. **Conclusion:** A different cost function is needed for logistic regression to ensure it is convex and that gradient descent can work reliably.

---

### **Logistic Loss Function**

To build a better cost function, we first define the error on a _single_ training example. This is called the **Loss Function**, denoted as $L(f_{\vec{w},b}(\vec{x}), y)$.

The **Logistic Loss Function** is defined as follows:

- If the true label **$y=1$**:
  $$L(f(\vec{x}), 1) = -\log(f(\vec{x}))$$
- If the true label **$y=0$**:
  $$L(f(\vec{x}), 0) = -\log(1 - f(\vec{x}))$$

---

### **Intuition Behind the Logistic Loss 🧠**

This loss function is designed to heavily penalize predictions that are confidently wrong.

- **When the true label is 1 ($y=1$):**

  - If the model predicts $f(\vec{x}) \to 1$ (a correct, confident prediction), the loss $-\log(1)$ approaches **0**.
  - If the model predicts $f(\vec{x}) \to 0$ (an incorrect, confident prediction), the loss $-\log(0)$ approaches **infinity**.

- **When the true label is 0 ($y=0$):**
  - If the model predicts $f(\vec{x}) \to 0$ (a correct, confident prediction), the loss $-\log(1-0)$ approaches **0**.
  - If the model predicts $f(\vec{x}) \to 1$ (an incorrect, confident prediction), the loss $-\log(1-1)$ approaches **infinity**.

---

### **The Full Cost Function**

The overall **cost function** $J(\vec{w},b)$ is simply the average of the loss function across all $m$ training examples:

$$J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})$$

- This new cost function is **convex**, which guarantees that gradient descent can converge to the global minimum.
