## **Regularized Logistic Regression**

Just like linear regression, logistic regression can be regularized to prevent overfitting, especially when using a large number of features or high-order polynomial terms.

---

### **The Problem: Overfitting in Logistic Regression**

When logistic regression is trained with many features, it can create an overly complex decision boundary that fits the training data perfectly but fails to generalize to new data.

---

### **The Regularized Cost Function**

To address overfitting, we add a regularization term to the logistic regression cost function. This penalizes large values for the weight parameters ($w_j$).

- **The final cost function for regularized logistic regression is:**
  $$J(\vec{w},b) = \left[ -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- Adding this term encourages the algorithm to find smaller parameter values, resulting in a simpler, smoother decision boundary that is less likely to overfit.

---

### **Gradient Descent Implementation 👨‍💻**

The gradient descent algorithm is updated to account for the new regularization term in the cost function.

- **Partial Derivatives:**

  - The derivative with respect to **$w_j$** gets an additional term:
    $$\frac{\partial J}{\partial w_j} = \left( \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m}w_j$$
  - The derivative with respect to **$b$** remains **unchanged**, as the bias term is not regularized.

- **Simultaneous Update Rules:**

  - **Update for $w_j$:**
    $$w_j := w_j - \alpha \left[ \left( \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}w_j \right]$$
  - **Update for $b$:**
    $$b := b - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) \right]$$

- **Key Similarity:** The update rule equations are **identical** to those for regularized linear regression. The only difference is the definition of **$f_{\vec{w},b}(\vec{x})$**, which is the sigmoid function for logistic regression.
