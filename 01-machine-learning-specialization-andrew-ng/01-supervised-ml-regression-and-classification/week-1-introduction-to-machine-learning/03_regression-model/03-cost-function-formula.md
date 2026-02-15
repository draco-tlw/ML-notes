## **The Cost Function**

The **cost function** is a crucial component of a learning algorithm. It provides a way to measure how well the model is performing by quantifying the error between its predictions and the actual true values in the training data. The goal of training is to find the model parameters that **minimize** this cost.

---

### **Model Parameters (w and b)**

- In the linear model $f_{w,b}(x) = wx + b$, the variables **`w`** and **`b`** are called the **parameters** of the model (also known as **weights** or **coefficients**).
- These are the values the learning algorithm will adjust during training to find the best-fit line for the data.
- **`b`** is the **y-intercept**, determining where the line crosses the vertical axis.
- **`w`** is the **slope**, determining the steepness of the line.

---

### **Constructing the Cost Function 🔨**

The goal is to find parameters `w` and `b` so that the model's predictions, $\hat{y}^{(i)}$, are as close as possible to the true target values, $y^{(i)}$, for all training examples. The cost function measures the total error. It's constructed in a few steps:

1. **Calculate the Error:** For each training example `i`, the error is the difference between the prediction and the true value:
   $$\text{error} = \hat{y}^{(i)} - y^{(i)} = f_{w,b}(x^{(i)}) - y^{(i)}$$

2. **Calculate the Squared Error:** We square the error to make it always positive and to penalize larger errors more heavily.
   $$(\text{error})^2 = (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

3. **Sum the Squared Errors:** We add up the squared errors for all `m` training examples to get a total error for the entire dataset.
   $$\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

4. **Calculate the Average (Mean) Squared Error:** To make the cost independent of the dataset size, we take the average. By convention, we divide by **`2m`** instead of just `m`. The division by 2 is a mathematical convenience that simplifies calculations for gradient descent later on.

---

### **The Squared Error Cost Function Formula**

The final formula for the cost function, denoted as **$J(w,b)$**, is:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

- This is known as the **Squared Error Cost Function**, and it is the most common cost function used for linear regression problems.
- Our goal is to find the values of `w` and `b` that result in the **smallest possible value** for $J(w,b)$.
