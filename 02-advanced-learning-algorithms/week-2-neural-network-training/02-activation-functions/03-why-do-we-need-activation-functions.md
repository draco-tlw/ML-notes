### **Why Non-Linear Activation Functions are Essential**

A neural network **must** use **non-linear activation functions** (like ReLU or sigmoid) in its hidden layers.

If a neural network only uses **linear activation functions** ($g(z) = z$) in all its layers, the entire network becomes computationally equivalent to a simple **linear regression** model. This completely defeats the purpose of having multiple "deep" layers, as the network will be unable to learn complex, non-linear patterns.

---

### **Mathematical Proof (The "Collapse") ⛓️**

Let's examine a simple 2-layer network where every activation function is linear ($g(z) = z$).

1.  **Layer 1 (Hidden Layer):**
    - The computation is: $a_1 = g(w_1x + b_1)$
    - With a linear activation, this becomes: **$a_1 = w_1x + b_1$**

2.  **Layer 2 (Output Layer):**
    - The computation is: $a_2 = g(w_2a_1 + b_2)$
    - With a linear activation, this becomes: **$a_2 = w_2a_1 + b_2$**

3.  **Substitute $a_1$ into $a_2$:**
    - $a_2 = w_2(w_1x + b_1) + b_2$
    - $a_2 = (w_2w_1)x + (w_2b_1 + b_2)$

4.  **Result:**
    - We can define a new, single $W = w_2w_1$ and a new, single $B = w_2b_1 + b_2$.
    - The final output is: **$a_2 = Wx + B$**

This final equation is just the formula for simple linear regression. No matter how many "deep" layers you add, if they are all linear, they "collapse" into a single linear function.

**Key Principle:** A linear function of another linear function is, itself, just a linear function.

---

### **What About Mixed Functions?**

This "collapse" still happens even if the output layer is non-linear (like sigmoid).

- If all **hidden layers are linear** ($g(z)=z$) but the **output layer is sigmoid**:
- The network collapses and becomes equivalent to a simple **logistic regression** model.
- You gain no advantage from the "deep" architecture.

---

### **Conclusion & Rule of Thumb 📜**

- **Hidden Layers:** You **must** use a non-linear activation function.
  - **Recommendation:** The default and most common choice is **ReLU**.
- **Output Layer:** You _can_ use a linear activation function, but **only** if you are solving a regression problem (as discussed in the previous video).
