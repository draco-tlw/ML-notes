### **The Problem: How to Choose a Model?**

The test set ($J_{test}$) gives us a good estimate of how well a _single_ model generalizes to new data. But what if we are trying to decide _which_ model to use in the first place?

For example, we might be choosing between 10 different models:

- Model 1: 1st-degree polynomial ($d=1$)
- Model 2: 2nd-degree polynomial ($d=2$)
- ...
- Model 10: 10th-degree polynomial ($d=10$)

---

### **A Flawed Procedure 🚫**

A tempting, but flawed, procedure would be:

1.  Train all 10 models on the **training set**.
2.  Evaluate all 10 models on the **test set**.
3.  See which model had the lowest test error (e.g., $d=5$).
4.  Report this lowest test error as our generalization error.

**Why this is flawed:** We have now used the test set to make a decision—we "fit" the parameter `d` (the degree of the polynomial) to our test set. By picking the _best_ result from this set, our reported test error ($J_{test}$) is now an **overly optimistic** (artificially low) estimate of how the model will do on new, unseen data.

---

### **The Solution: The 3-Way Data Split 🧪**

To correctly perform model selection, we split our data into **three** separate sets.

1.  **Training Set (e.g., 60% of data)**
    - **Purpose:** To train the parameters ($w, b$) for _each_ of our candidate models.

2.  **Cross-Validation (CV) Set (e.g., 20% of data)**
    - **Purpose:** To select the best model. We use this set to "cross-check" the performance of the different models.
    - Also known as the **validation set** or **development (dev) set**.

3.  **Test Set (e.g., 20% of data)**
    - **Purpose:** To provide a final, **unbiased estimate of the generalization error** of our _one_ chosen model. This set is "untouched" until the very end.

---

### **The Correct Model Selection Procedure ⚙️**

Here is the standard, correct procedure for choosing the best model and evaluating it fairly:

1.  **Train all candidate models** (e.g., all 10 polynomial models) on the **training set**.
2.  **Evaluate all trained models** on the **cross-validation (CV) set**.
3.  **Pick the one model** that performed the best (had the lowest error) on the **CV set** ($J_{cv}$). For example, let's say the $d=4$ polynomial had the lowest CV error.
4.  **Finally, take your single chosen model** (the $d=4$ model) and evaluate it _one time_ on the **test set**.
5.  Report the resulting $J_{test}$ as the unbiased estimate of your model's generalization error.

This same procedure is used for choosing any model parameters, such as the architecture of a neural network (e.g., comparing a small, medium, and large network to see which performs best on the CV set).

**Best Practice:** All decisions—fitting parameters ($w, b$), choosing the model (polynomial degree `d`, network architecture, regularization $\lambda$)—should be made using _only_ the **training set** and **cross-validation set**. The **test set** should only be used once at the very end to get a final performance metric.
