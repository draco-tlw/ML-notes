### **Learning Curves 📈**

A **learning curve** is a diagnostic tool that plots your model's **error** (both $J_{train}$ and $J_{cv}$) as a function of the **training set size** (`m_train`).

It helps you understand how your algorithm's performance changes as it gets more "experience" (more data).

- **Horizontal Axis:** `m_train` (number of training examples).
- **Vertical Axis:** Error (e.g., squared error or misclassification error).

---

### **The Shape of a "Good" Learning Curve**

For a well-fitting model (one that is "just right"), the learning curves have a predictable shape:

- **Training Error ($J_{train}$):**
  - When `m_train` is very small (e.g., 1 or 2 examples), the error is **zero or near-zero** because it's easy for the model to perfectly fit a tiny amount of data.
  - As `m_train` **increases**, it becomes harder for the model to perfectly fit every example, so **$J_{train}$ increases** and then flattens out.

- **Cross-Validation Error ($J_{cv}$):**
  - When `m_train` is very small, the model is trained on tiny, unrepresentative data, so it generalizes terribly. $J_{cv}$ is **very high**.
  - As `m_train` **increases**, the model learns from more data and generalizes better, so **$J_{cv}$ decreases** and then flattens out.

**Key Property:** For a good model, $J_{cv}$ is typically higher than $J_{train}$, but the two curves should converge to a low error value as `m_train` gets large.

---

### **Using Learning Curves to Diagnose Bias & Variance**

Learning curves have very different shapes depending on whether your model has a bias or variance problem.

#### **1. High Bias (Underfitting)**

- **What it looks like:** The $J_{train}$ and $J_{cv}$ curves are **both high** and **close to each other**. They flatten out at a high error level.
- **Why it happens:** The model is too simple (like fitting a straight line to a curve). Even with very little data, $J_{train}$ is high because the line can't fit the data. As you add more data, the line doesn't change much, so both errors remain high.
- **Key Takeaway:** If an algorithm has high bias, **getting more training data will not help.** The curves will just stay flat at a high error level.

#### **2. High Variance (Overfitting)**

- **What it looks like:** There is a **large gap** between the $J_{cv}$ (which is high) and $J_{train}$ (which is low).
- **Why it happens:** The model is too complex (like a 10th-order polynomial). It perfectly fits the training data (low $J_{train}$) but fails to generalize (high $J_{cv}$).
- **Key Takeaway:** If an algorithm has high variance, **getting more training data is likely to help.** As `m_train` increases, $J_{train}$ will rise, but $J_{cv}$ will fall, and the gap between them will shrink.

---

### **Practical Use**

- Plotting a full learning curve can be computationally expensive (you have to train many models on different subsets of data).
- However, the _concept_ is a powerful mental model. The main way to diagnose bias and variance is still to look at the gap between your baseline, $J_{train}$, and $J_{cv}$ using all your data.
