## **Regression Trees (Optional) 🌳**

Decision trees can be generalized to solve **regression problems**, where the goal is to predict a continuous numerical value (like a price or, in this case, an animal's **weight**) instead of a discrete class (like "Cat" or "Not Cat").

---

### **1. How Regression Trees Make Predictions**

The structure of the tree (splitting on features like "Ear Shape" or "Face Shape") is built in the same way. The only difference is the value stored in the **leaf nodes**.

- **Classification Tree Leaf:** Predicts the **majority class** (e.g., "Cat") of the training examples that fall into that leaf.
- **Regression Tree Leaf:** Predicts the **average value** of the target `y` (e.g., the average weight) of all the training examples that fall into that leaf.

**Example:**
If 4 animals (with weights 7.2, 7.6, 8.4, 10.2) end up in a leaf, the prediction for that leaf becomes their average: **8.35 lbs**.

---

### **2. How Regression Trees are Trained: Reducing Variance**

To build a regression tree, we need a different metric to measure "impurity." For classification, we minimized **entropy**. For regression, we minimize **variance**.

- **Entropy (Classification):** Measures the _impurity_ of classes (how mixed "Cat" and "Not Cat" are).
- **Variance (Regression):** Measures the _impurity_ of numbers (how "spread out" the weights are).
  - A set of numbers like `[8.8, 15, 11, 18, 20]` has **high variance**.
  - A set of numbers like `[7.2, 7.6, 8.4, 10.2]` has **low variance**.
  - The goal is to create splits that result in nodes with the lowest possible variance.

---

### **Finding the Best Split 🧮**

The algorithm's goal is to find the feature split that results in the **largest reduction in variance**. This is the direct equivalent of seeking the highest "information gain" in a classification tree.

The process at each node is:

1.  **Calculate Parent Variance:** First, calculate the variance of all `y` values (weights) in the current (parent) node.
    - _Example:_ Variance of all 10 animals = **20.51**

2.  **Test Each Feature Split:** For each possible feature (e.g., "Ear Shape"):
    - **a.** Split the data and calculate the variance for the left and right child nodes (e.g., `Variance_left = 1.47`, `Variance_right = 21.87`).
    - **b.** Calculate the **weighted average variance** of the children.
      - `avg_variance = (w_left * Variance_left) + (w_right * Variance_right)`
      - (e.g., `(5/10 * 1.47) + (5/10 * 21.87) = 11.67`)

3.  **Calculate Variance Reduction:**
    - `Variance Reduction = Parent Variance - Weighted Average Variance`
    - _Example (Ear Shape):_ `20.51 - 11.67 = 8.84`
    - _Example (Face Shape):_ `20.51 - 19.87 = 0.64`
    - _Example (Whiskers):_ `20.51 - 14.29 = 6.22`

4.  **Decision:**
    - The "Ear Shape" split gives the **highest variance reduction** (8.84).
    - Therefore, the algorithm chooses "Ear Shape" as the feature for this split.

This recursive process continues until a stopping criterion (like `max_depth` or minimum examples) is met.
