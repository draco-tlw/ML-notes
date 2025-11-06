### **The Gradient Descent Algorithm ⚙️**

Gradient descent is an iterative process that repeatedly updates the model parameters `w` and `b` to minimize the cost function $J(w,b)$.

The core of the algorithm is the following update rule, which is repeated until the model converges:

$$w := w - \alpha \frac{\partial}{\partial w} J(w,b)$$
$$b := b - \alpha \frac{\partial}{\partial b} J(w,b)$$

---

### **Components of the Update Rule 🧐**

Let's break down what each part of the rule means:

- **`:=` (The Assignment Operator):** In this context, the equal sign means we are **assigning** a new value to the parameter. We calculate the entire expression on the right and use it to update the value of `w` or `b` on the left.

- **`α` (Alpha, The Learning Rate):**

  - Alpha is a small positive number (e.g., 0.01) that controls **how big of a step** you take downhill on each iteration.
  - A large `α` means taking aggressive, large steps.
  - A small `α` means taking small, cautious baby steps.
  - Choosing a good value for `α` is a critical part of making the algorithm work well.

- **$\frac{\partial}{\partial w} J(w,b)$ (The Derivative Term):**
  - You don't need to know calculus to understand its role here.
  - This term calculates the **direction of the steepest descent**. It tells the algorithm _which way_ to go to reduce the cost most effectively.
  - Multiplying the direction (derivative) by the step size (`α`) determines the overall update for the parameter.

---

### **Correct Implementation: Simultaneous Updates ✅**

A crucial detail for implementing gradient descent correctly is to perform **simultaneous updates** for all parameters. This means you must calculate the new values for `w` and `b` _before_ updating either of them.

- **Correct Method:**

  1. Calculate the update for `w` and store it in a temporary variable: `temp_w = w - ...`
  2. Calculate the update for `b` and store it in another temporary variable: `temp_b = b - ...`
  3. Update `w` and `b` with the values from the temporary variables: `w = temp_w`, `b = temp_b`.

- **Incorrect Method (Sequential Update) ❌:**
  1. Calculate the new `w` and immediately update it.
  2. Use this **newly updated `w`** to calculate the update for `b`.
  - This is technically a different algorithm and is not the standard implementation of gradient descent.
