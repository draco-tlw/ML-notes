## **Types of Supervised Learning: Classification**

**Classification** is the second major type of supervised learning.

- **Goal:** To predict a **discrete category** (or class) from a small, finite set of possible outputs.

- **Key Difference from Regression:**
  - **Regression** predicts a continuous number from an infinite range (e.g., a house price).
  - **Classification** predicts one of a few distinct outcomes (e.g., is this email spam or not?).

---

### **Example: Breast Cancer Detection 🩺**

- **Input (x):** Features of a tumor, such as its **size**.
- **Output (y):** A category — either **Malignant** (cancerous) or **Benign** (not cancerous).
- **Data Representation:** These categories are often mapped to numbers for the model to process, for example:
  - Benign = **0**
  - Malignant = **1**

---

### **Multi-Class Classification**

- Classification is not limited to just two categories. A problem can have multiple possible output classes.
- For example, a model could be trained to distinguish between "Benign," "Cancer Type 1," and "Cancer Type 2."
- In this case, the output `y` could be 0, 1, or 2, each representing one of the distinct classes.

---

### **Using Multiple Features & The Decision Boundary**

- Models can use multiple input features to make more accurate predictions.
- **Example:** Using both a patient's **age** and the **tumor size**.
- When using multiple features, the learning algorithm's goal is to find a **decision boundary**—a line or curve that best separates the different classes in the data.
- For a new patient, the model determines which side of this boundary their data point falls on to make its classification.

---

### **Summary: Regression vs. Classification**

| Feature         | **Regression**          | **Classification**          |
| :-------------- | :---------------------- | :-------------------------- |
| **Output Type** | Continuous (a number)   | Discrete (a category/class) |
| **Goal**        | Predict a quantity      | Predict a label             |
| **Example**     | Predicting house prices | Identifying spam emails     |
