## **The Decision Boundary 🗺️**

The decision boundary is a key concept in understanding how a logistic regression model makes its final predictions. It is the line or curve that separates the different predicted classes.

---

### **From Probability to Prediction**

To convert the model's probability output into a discrete prediction (0 or 1), a **threshold** is used.

- The model's output is the probability that $y=1$:
  $$f_{\vec{w},b}(\vec{x}) = P(y=1 | \vec{x})$$
- A common threshold is **0.5**.
- **Prediction Rule:**
  - If $f_{\vec{w},b}(\vec{x}) \ge 0.5$, predict $\hat{y} = 1$.
  - If $f_{\vec{w},b}(\vec{x}) < 0.5$, predict $\hat{y} = 0$.

By analyzing the Sigmoid function, $g(z)$, we know that $g(z) \ge 0.5$ only when its input, $z$, is greater than or equal to 0.

- This simplifies the prediction rule significantly:
  - Predict $\hat{y} = 1$ when $z = \vec{w} \cdot \vec{x} + b \ge 0$.
  - Predict $\hat{y} = 0$ when $z = \vec{w} \cdot \vec{x} + b < 0$.

The **decision boundary** is the line defined by the equation where the model is exactly on the fence:
$$\vec{w} \cdot \vec{x} + b = 0$$

---

### **Types of Decision Boundaries**

#### **Linear Decision Boundary**

- When using simple features (e.g., $x_1, x_2$), the decision boundary is a straight line.
- **Example:**
  - Model with two features: $z = w_1x_1 + w_2x_2 + b$.
  - If parameters are $w_1=1, w_2=1, b=-3$.
  - The decision boundary is the line where $x_1 + x_2 - 3 = 0$, or **$x_1 + x_2 = 3$**.

#### **Non-Linear Decision Boundary**

- By using **polynomial features**, logistic regression can learn complex, non-linear decision boundaries.
- **Example:**
  - Model with polynomial features: $z = w_1x_1^2 + w_2x_2^2 + b$.
  - If parameters are $w_1=1, w_2=1, b=-1$.
  - The decision boundary is the curve where $x_1^2 + x_2^2 - 1 = 0$, or **$x_1^2 + x_2^2 = 1$**. This is a circle.
- Higher-order polynomial terms can create even more complex shapes, like ellipses or other intricate curves, allowing the model to fit very complex datasets.
