## **Introduction to Derivatives (Optional)**

This optional section introduces the mathematical concept of a **derivative**, which is the core component of **backpropagation**. Backpropagation is the algorithm that TensorFlow uses to automatically calculate the derivatives needed for gradient descent (or the Adam optimizer).

This lesson uses an intuitive, non-calculus-based approach to understand what a derivative is.

---

## **An Intuitive Definition of a Derivative 💡**

A derivative measures the **sensitivity** or **effect** that a tiny change in one variable has on another.

For a cost function $J(w)$, the derivative of $J$ with respect to $w$ (written as $\frac{\partial J}{\partial w}$) answers the question: "If I nudge $w$ by a tiny amount, how much does $J$ change in response?"

### The "Epsilon Nudge" Method

Let's use a simple cost function: $J(w) = w^2$

1.  **Start with a value:** Let **$w = 3$**.
    - The cost $J(3) = 3^2 = 9$.

2.  **Nudge the variable:** Increase $w$ by a tiny amount, $\epsilon$ (epsilon). Let's say **$\epsilon = 0.001$**.
    - Our new $w$ is $3 + 0.001 = 3.001$.

3.  **Calculate the new cost:**
    - $J(3.001) = (3.001)^2 = 9.006001$.

4.  **Analyze the change:**
    - We increased $w$ by **$0.001$** ($\epsilon$).
    - The cost $J$ increased by **$0.006001$**.
    - This change in $J$ ($0.006001$) is almost exactly **6** times the change in $w$ ($0.001$).

5.  **Conclusion:** The **derivative is 6**.
    - **Informal Definition:** If $w$ goes up by $\epsilon$, and $J(w)$ goes up by $k \times \epsilon$, then the derivative is $k$.

---

## **How Derivatives Power Gradient Descent**

The gradient descent update rule is: $w := w - \alpha \frac{\partial J}{\partial w}$

The derivative ($\frac{\partial J}{\partial w}$) tells the algorithm how big of a step to take.

- **Large Derivative:** If the derivative is large (like 6), it means changing $w$ has a **big effect** on the cost $J$. Therefore, gradient descent should take a **large step** to get a big improvement.
- **Small Derivative:** If the derivative is small, changing $w$ has a **small effect** on $J$. Therefore, gradient descent should take a **small step**.

---

## **Key Properties of Derivatives**

### 1. The Derivative Depends on the Value of `w`

The derivative is not a fixed number; it changes depending on the _current value_ of $w$. For our function $J(w) = w^2$:

- If **$w = 3$**, $\frac{\partial J}{\partial w} = 6$.
- If **$w = 2$**, $\frac{\partial J}{\partial w} = 4$. (A 0.001 nudge makes $J$ go from 4 to ~4.004).
- If **$w = -3$**, $\frac{\partial J}{\partial w} = -6$. (A +0.001 nudge to $w$ makes $J$ _decrease_ by 0.006, so the change is $-6 \times \epsilon$).

### 2. The Calculus Connection (Optional)

- **Slope:** In calculus, the derivative is formally defined as the **slope of the line tangent** to the function at that exact point.
- **Formula:** For $J(w) = w^2$, the derivative formula is $\frac{\partial J}{\partial w} = 2w$.
- This formula perfectly matches our intuitive findings:
  - At $w=3$, derivative = $2 \times 3 = 6$.
  - At $w=2$, derivative = $2 \times 2 = 4$.
  - At $w=-3$, derivative = $2 \times (-3) = -6$.

### 3. Examples of Other Derivatives

The video uses the Python library `SymPy` to find the derivative formulas for other functions:

| Function ($J(w)$) | Derivative ($\frac{\partial J}{\partial w}$) | Value at $w=2$ |
| :---------------- | :------------------------------------------- | :------------- |
| $w^2$             | $2w$                                         | 4              |
| $w^3$             | $3w^2$                                       | 12             |
| $w$               | $1$                                          | 1              |
| $1/w$             | $-1/w^2$                                     | -0.25          |

### 4. A Note on Notation

- You may see two symbols for derivatives: $d/dw$ (total derivative) and $\partial/\partial w$ (partial derivative).
- The **"squiggly d" ($\partial$)** is used when a function (like our cost $J$) depends on _more than one variable_ (e.g., $w_1, w_2, \dots, b_1, b_2, \dots$).
- Since our cost function _always_ depends on many parameters, this course will **always use the partial derivative symbol $\partial/\partial w$**.
