## **Simplified Cost Function for Logistic Regression**

To make implementation easier, the two-part logistic loss function can be combined into a single, equivalent equation.

---

### **Simplified Loss Function**

Recall the original loss function was defined in two parts, one for when $y=1$ and another for when $y=0$. Because $y$ in a binary classification problem can _only_ be 0 or 1, we can combine these into one expression.

- **The simplified loss function** for a single training example is:
  $$L(f(\vec{x}), y) = -y \log(f(\vec{x})) - (1-y) \log(1 - f(\vec{x}))$$

- **How it Works:**
  - **Case 1: True label is 1 ($y=1$)**
    - The second term, $(1-y)\log(\dots)$, becomes $0 \cdot \log(\dots) = 0$.
    - The equation simplifies to $L = -1 \cdot \log(f(\vec{x}))$, which is the original loss for $y=1$.
  - **Case 2: True label is 0 ($y=0$)**
    - The first term, $-y\log(\dots)$, becomes $0 \cdot \log(\dots) = 0$.
    - The equation simplifies to $L = -(1-0) \log(1 - f(\vec{x}))$, which is the original loss for $y=0$.

This single-line formula is mathematically identical to the previous piecewise definition.

---

### **Full Cost Function using Simplified Loss**

The overall cost function, $J(\vec{w},b)$, is the average of this simplified loss function over all $m$ training examples.

- **Final Cost Function for Logistic Regression:**
  $$J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) \right]$$

- **Rationale for this Function:**
  - This cost function is derived from a statistical principle known as **Maximum Likelihood Estimation**.
  - Most importantly, it has the desirable property of being **convex**, which ensures that gradient descent can reliably find the global minimum.
