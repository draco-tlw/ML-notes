## **How a Supervised Learning Model Works**

The core process of supervised learning involves taking a dataset, feeding it into a learning algorithm, and producing a predictive model.

1. **Input (Training Set):** The process begins with a training set containing both the input features (`x`) and the correct output targets (`y`).
2. **Learning Algorithm:** The training set is fed into a learning algorithm.
3. **Output (The Model):** The learning algorithm's output is a **model**, which is a function, denoted as **`f`**.

---

### **The Model's Job: Making Predictions 🔮**

- The model `f` takes a new input `x` and produces an estimated output, called a **prediction**.
- **Prediction Notation:** The prediction is denoted by **$\hat{y}$** (pronounced "y-hat").
  - **`y`** represents the **true value** or target (from the training data).
  - **$\hat{y}$** represents the model's **estimated value** or prediction.

---

### **Representing the Model: Linear Regression**

A key step in designing a model is choosing how to represent the function `f`. For linear regression, we use the equation of a straight line.

- **The Model Function:**
  $$f_{w,b}(x) = wx + b$$

  - This is often shortened to **$f(x) = wx + b$**.
  - **`x`** is the input feature (e.g., house size).
  - **`w`** and **`b`** are the **parameters** of the model. `w` is the weight (slope of the line) and `b` is the bias (y-intercept). The learning algorithm's job is to find the best values for `w` and `b` to fit the data.

- **Why a Linear Model?** While more complex, non-linear functions (curves) exist, a linear model is a simple and effective foundation for understanding more complex models later.

- **Model Name:** This specific model is called **linear regression with one variable**, or **univariate linear regression**, because it uses a single input variable (`x`).
