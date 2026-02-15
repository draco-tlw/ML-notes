## **The Supervised Learning Process**

Supervised learning involves training a model on a dataset that contains the "right answers" (labels). The model learns from this labeled data and can then make predictions on new, unseen data.

---

### **Linear Regression: The First Model**

**Linear Regression** is one of the most widely used machine learning models. Its goal is to fit a **straight line** to a dataset to model the relationship between an input and an output.

- **Example: Housing Price Prediction**
  - **Goal:** Predict a house's price based on its size.
  - **Dataset:** A collection of houses with their known sizes (input) and selling prices (output).
  - **Process:**
    1. Plot the known data (size vs. price).
    2. The linear regression model fits the best possible straight line through these data points.
    3. To predict the price of a new house, you find its size on the horizontal axis, trace up to the line, and then across to the vertical axis to get the estimated price.

---

### **Regression vs. Classification (Recap)**

- **Regression Model:** Predicts a **continuous numerical value** from an infinite range of possibilities (e.g., predicting a price of $220,000).
- **Classification Model:** Predicts a **discrete category** from a small, finite set of possibilities (e.g., cat vs. dog, or disease vs. no disease).

---

### **Key Terminology & Notation ✍️**

This is the standard notation used in machine learning to describe a training dataset.

- **Training Set:** The dataset used to train the model.
- **Input Variable (Feature):** Denoted by **$x$**. This is the input data used for prediction (e.g., size of a house).
- **Output Variable (Target):** Denoted by **$y$**. This is the value the model is trying to predict (e.g., price of a house).
- **m:** The total **number of training examples** in the dataset (i.e., the number of rows in the data table).
- **$(x, y)$:** Represents a **single training example**—a single input-output pair.
- **$(x^{(i)}, y^{(i)})$**: Refers to the **$i$-th training example** in the dataset. The superscript `(i)` is an index pointing to the i-th row in the table, **it is not an exponent**.
  - For example, $(x^{(1)}, y^{(1)})$ is the first house in the dataset (first row of the data table).
