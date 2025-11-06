### **Week 3: Introduction to Classification**

- This week focuses on **classification**, a type of machine learning problem where the output variable, $y$, belongs to a small, finite set of possible values or categories.
- This is distinct from **linear regression**, which predicts a continuous numerical value.

---

### **Binary Classification**

- **Binary Classification** is a specific type of classification problem where there are only two possible output classes.

- **Examples:**

  - Spam detection (Spam / Not Spam)
  - Financial fraud detection (Fraudulent / Not Fraudulent)
  - Medical diagnosis (Malignant / Benign tumor)

- **Class Representation Conventions:**
  - The two classes are often represented as No/Yes, False/True, or most commonly, **0/1**.
  - **Negative Class:** The "absence" class, conventionally represented by **0** (e.g., not spam, benign tumor).
  - **Positive Class:** The "presence" class, conventionally represented by **1** (e.g., spam, malignant tumor).
  - _Note:_ The terms 'positive' and 'negative' are just labels and do not inherently imply 'good' or 'bad'.

---

### **Why Linear Regression Fails for Classification**

- A naive approach to classification could be to apply linear regression and use a threshold to determine the class.

  - For example, fit a straight line to the data.
  - Set a threshold (e.g., at 0.5 on the predicted y-axis).
  - If the model's output $f(x) \ge 0.5$, predict class 1.
  - If the model's output $f(x) < 0.5$, predict class 0.

- **The Problem with Outliers:**
  1. Linear regression tries to find the best-fitting line for _all_ data points.
  2. If an outlier data point is added (e.g., a very large tumor that is clearly malignant), it will pull the best-fit line towards it.
  3. This shift in the line can change the **decision boundary** (the point where the prediction switches from 0 to 1).
  4. The result is a less accurate model, as the classification threshold is now in a worse position for the original data points. Linear regression is too sensitive to such outliers in a classification context.

---

### **Introduction to Logistic Regression**

- **Logistic Regression** is a classification algorithm that addresses the shortcomings of using linear regression for this task.
- Despite its name, it is used for **classification**, not regression. The name is due to historical reasons.
- Its key advantage is that its output is always constrained between **0 and 1**, which is ideal for representing probabilities in classification.
