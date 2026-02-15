## **Regularization's Effect on Bias and Variance**

The **regularization parameter ($\lambda$)** is a critical tool for controlling the bias-variance trade-off in your model. By adjusting $\lambda$, you can directly influence your model's complexity, moving it along the spectrum from underfitting (high bias) to overfitting (high variance).

---

### **The Role of $\lambda$ in Model Fit**

We'll use a 4th-order polynomial model (which tends to overfit) to see the effects of $\lambda$.

- **When $\lambda$ is very large (e.g., 10,000):**
  - The cost function places an enormous penalty on all weight parameters ($w_j$).
  - To minimize the cost, the algorithm is forced to set all $w_j$ parameters to be very close to zero.
  - The model effectively becomes $f(x) \approx b$ (a flat horizontal line).
  - This model is too simple and **underfits** the data.
  - **Diagnosis:** **High Bias**. ($J_{train}$ will be high, and $J_{cv}$ will also be high).

- **When $\lambda$ is zero:**
  - The regularization term disappears entirely.
  - The model becomes an unregularized 4th-order polynomial, which will fit the training data perfectly (or near-perfectly) by creating a "wiggly" curve.
  - This model is too complex and **overfits** the data.
  - **Diagnosis:** **High Variance**. ($J_{train}$ will be low, but $J_{cv}$ will be much higher).

- **When $\lambda$ is "just right" (an intermediate value):**
  - The model finds a balance. It fits a smooth, reasonable curve that captures the data's trend without fitting the noise.
  - **Diagnosis:** **Low Bias, Low Variance**. ($J_{train}$ is low, and $J_{cv}$ is also low and close to $J_{train}$).

---

### **Visualizing Error vs. $\lambda$**

We can diagnose the model by plotting the training and cross-validation error as a function of $\lambda$.

- **Horizontal Axis:** The value of $\lambda$.
- **Vertical Axis:** The error ($J$).

#### **Curve Behavior**

- **Training Error ($J_{train}$):**
  - When $\lambda$ is 0, the model overfits, so $J_{train}$ is **very low**.
  - As $\lambda$ **increases**, the penalty on the weights gets stronger, forcing the model to be simpler and _worse_ at fitting the training data.
  - Therefore, **$J_{train}$ consistently increases as $\lambda$ increases**.

- **Cross-Validation Error ($J_{cv}$):**
  - This curve has a **U-shape**.
  - **Region 1: Low $\lambda$ (e.g., $\lambda=0$):** The model is overfitting. $J_{train}$ is low, but $J_{cv}$ is **high**. This is the **high variance** region.
  - **Region 2: High $\lambda$ (e.g., $\lambda=10,000$):** The model is underfitting. $J_{train}$ is high, and $J_{cv}$ is also **high**. This is the **high bias** region.
  - **The "Sweet Spot":** The minimum of the $J_{cv}$ curve represents the optimal value for $\lambda$, which provides the best balance between bias and variance.

> **Note on Plots:** This $\lambda$ vs. Error plot looks like a "mirror image" of the _Model Complexity (d) vs. Error_ plot.
>
> - For **degree `d`**: High Bias is on the left (simple models), High Variance is on the right (complex models).
> - For **lambda $\lambda$**: High Variance is on the left (complex models), High Bias is on the right (simple models).

---

### **Choosing the Optimal $\lambda$ (Model Selection Procedure)**

You can use your cross-validation set to automatically find the best value for $\lambda$.

1.  **Create a list of $\lambda$ values** to try (e.g., `0`, `0.01`, `0.02`, `0.04`, `0.08`, `0.16`, ..., `10`).
2.  **Iterate through the list:** For each value of $\lambda$:
    a. Train your model on the **training set** to find the corresponding parameters $w, b$.
    b. Evaluate the trained model on the **cross-validation set** to get its error, $J_{cv}$.
3.  **Select the best $\lambda$:** Choose the value of $\lambda$ that resulted in the **lowest $J_{cv}$**.
4.  **Report final performance:** Evaluate the model (with your chosen $\lambda$) on the separate **test set** ($J_{test}$) to get an unbiased estimate of its generalization error.
