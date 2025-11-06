## **Gradient Descent for Multiple Linear Regression**

This section combines the concepts of multiple linear regression and vectorization to create an efficient implementation of the gradient descent algorithm.

### **Review: Vector Notation ✍️**

To simplify our equations, we group our parameters and features into **vectors**.

- **Parameters:**
  - Instead of separate parameters $w_1, w_2, \dots, w_n$, we define a single parameter **vector** $\vec{w} = [w_1, w_2, \dots, w_n]$.
  - `b` remains a single number (a scalar).
- **Model:** The model can be written compactly using the **dot product**:
  $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
- **Cost Function:** The cost function is now a function of the vector $\vec{w}$ and the scalar `b`:
  $$J(\vec{w}, b)$$

---

### **The Gradient Descent Algorithm**

The algorithm updates all `n` of the `w` parameters and the `b` parameter simultaneously on each iteration.

- **Update Rule for $w_j$ (for j=1...n):**
  $$w_j := w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w},b)$$
- **Update Rule for $b$:**
  $$b := b - \alpha \frac{\partial}{\partial b} J(\vec{w},b)$$

---

### **Derivative Formulas for Multiple Regression**

The specific derivative terms for the multiple linear regression cost function are very similar to the univariate case.

- **Derivative with respect to $w_j$:**
  $$\frac{\partial}{\partial w_j} J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)}$$
- **Derivative with respect to $b$:**
  $$\frac{\partial}{\partial b} J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})$$

The main difference from the single-feature model is that we now have a separate update for each feature's weight ($w_1, w_2$, etc.), and the derivative for $w_j$ specifically uses the corresponding feature value, $x_j$.

---

### **Alternative Method: The Normal Equation (Side Note) 📝**

- For linear regression _only_, there is a non-iterative method for finding the optimal `w` and `b` called the **Normal Equation**.
- It involves solving for the parameters directly using advanced linear algebra, without needing to choose a learning rate `α` or run multiple iterations.
- **Disadvantages:**
  - It **does not generalize** to other algorithms like logistic regression or neural networks.
  - It can be very **slow** if the number of features (`n`) is large.
- **Conclusion:** Gradient descent is a more general and often more practical method for most machine learning applications.
