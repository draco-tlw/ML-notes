## **Running Gradient Descent: The Algorithm in Action**

This section visualizes the step-by-step process of gradient descent finding the optimal parameters for a linear regression model.

### **The Iterative Process Visualized 📉**

1.  **Initialization:** We start with an initial guess for the parameters `w` and `b`. This corresponds to a specific starting point on the cost function's contour plot and a poorly fitting line on the data plot.

2.  **Taking Steps:**

    - With each iteration (or step) of gradient descent, the algorithm calculates the direction of steepest descent and updates `w` and `b`.
    - On the contour plot, you can see the point representing `(w, b)` move closer to the center (the minimum).
    - Simultaneously, on the data plot, the straight line `f(x)` adjusts its slope and intercept, providing a slightly better fit to the data with each step.

3.  **Convergence:**
    - This process is repeated, with the cost `J(w,b)` decreasing at each step.
    - Eventually, the algorithm **converges** at the **global minimum**—the center of the contour plot.
    - The final `w` and `b` values at this minimum define the model that best fits the training data. This final model can then be used to make predictions on new data.

---

### **Batch Gradient Descent**

- The specific version of the algorithm we have been using is called **Batch Gradient Descent**.
- **Definition:** The term "batch" means that on **every single step** of gradient descent, we look at the **entire training set** (all `m` examples) to calculate the derivatives. This is reflected in the summation term, $\sum_{i=1}^{m}$, which runs over all the examples.
- _Note:_ Other versions of gradient descent exist that look at smaller subsets of data, but for now, we are using the "batch" approach.

---
