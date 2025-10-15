## **Gradient Descent**

**Gradient descent** is an optimization algorithm used to find the minimum of a function. In machine learning, it's used to find the values for the model's **parameters** (e.g., `w` and `b`) that result in the lowest possible cost. It's a foundational algorithm used to train everything from simple linear regression models to advanced neural networks.

---

### **The Goal: Minimizing the Cost Function**

The objective of gradient descent is to systematically find the parameters `w` and `b` that **minimize** the cost function $J(w,b)$.

The algorithm works for any cost function, including more complex ones with many parameters ($w_1, w_2, ..., w_n, b$).

---

### **How it Works: The Downhill Analogy 🚶‍♀️**

Imagine the cost function $J$ as a hilly landscape, where the height of the ground represents the cost. The goal is to find the bottom of a valley.

1. **Start Somewhere:** Begin by making an initial guess for the parameters `w` and `b`. A common starting point is to set them all to zero. This places you at a specific point on the hilly landscape.

2. **Look Around:** From your current position, look around 360 degrees and determine which direction is the **steepest path downhill**.

3. **Take a Step:** Take a small step in that direction of **steepest descent**. You are now at a new, slightly lower point on the hill.

4. **Repeat:** Continue repeating this process—look for the steepest path downhill and take another small step.

By repeatedly taking small steps in the steepest downhill direction, you will eventually converge to the bottom of a valley, which represents a minimum of the cost function.

---

### **An Important Property: Local Minima**

For complex, non-bowl-shaped cost functions (like those for neural networks), there can be multiple valleys, known as **local minima**.

- The specific local minimum that the algorithm finds **depends on the starting point**.
- Starting at a slightly different initial position can cause the algorithm to descend into a completely different valley.
- _Note:_ The squared error cost function for linear regression is a simple **bowl shape** (convex), so it only has **one global minimum**. This means that no matter where you start, gradient descent will always lead to the same optimal solution.
