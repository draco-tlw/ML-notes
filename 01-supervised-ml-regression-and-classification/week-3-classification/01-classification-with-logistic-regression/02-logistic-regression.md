### **The Logistic Regression Model**

- **Logistic Regression** is one of the most widely used classification algorithms.
- Unlike linear regression, which fits a straight line, logistic regression fits an **S-shaped curve** to the data. This curve is more suitable for classification tasks where the output is either 0 or 1.

---

### **The Sigmoid (Logistic) Function**

- The core of the logistic regression model is the **Sigmoid function**, denoted as $g(z)$.
- **Formula:**
  $$g(z) = \frac{1}{1 + e^{-z}}$$
  where `e` is the mathematical constant ≈ 2.718.

- **Properties of the Sigmoid Function:**
  - It takes any real number `z` as input and outputs a value between **0 and 1**.
  - If `z` is a large positive number, $e^{-z}$ approaches 0, so $g(z)$ approaches 1.
  - If `z` is a large negative number, $e^{-z}$ becomes very large, so $g(z)$ approaches 0.
  - When $z=0$, $g(0) = \frac{1}{1+e^0} = \frac{1}{1+1} = 0.5$.

---

### **Constructing the Model**

The logistic regression model is built in two steps:

1.  **Linear Combination:** First, calculate `z` using the same linear function as in regression:
    $$z = \vec{w} \cdot \vec{x} + b$$
2.  **Apply Sigmoid Function:** Pass the result `z` through the sigmoid function to get the final prediction:
    $$f_{\vec{w},b}(\vec{x}) = g(z) = g(\vec{w} \cdot \vec{x} + b)$$

- **Full Model Equation:**
  $$f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

---

### **Interpreting the Output 🧐**

- The output of the logistic regression model, $f_{\vec{w},b}(\vec{x})$, is interpreted as the **probability** that the output label $y$ is equal to 1, given the input features $\vec{x}$.
- **Formal Notation:**
  $$f_{\vec{w},b}(\vec{x}) = P(y=1 | \vec{x}; \vec{w}, b)$$

- **Example:**

  - If the model outputs `0.7` for a given tumor size, it means there is a **70% probability** that the tumor is malignant ($y=1$).
  - Since the probabilities must sum to 1, the probability of the tumor being benign ($y=0$) is:
    $$P(y=0) = 1 - P(y=1) = 1 - 0.7 = 0.3 \quad (30\% \text{ probability})$$

- The next step in using the model is to define a **decision boundary**, which determines how to map these probability outputs (e.g., 0.7) to a final, discrete class prediction (0 or 1).
