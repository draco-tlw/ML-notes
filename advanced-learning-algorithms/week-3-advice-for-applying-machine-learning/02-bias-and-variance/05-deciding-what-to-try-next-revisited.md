### **Deciding What to Try Next: An Action Plan 🗺️**

When your machine learning model has unacceptably large errors, the bias/variance diagnosis provides a clear guide on what to do. The six most common actions to take each map to fixing either high bias or high variance.

---

### **Fixing a High Variance Problem (Overfitting)**

If your diagnosis is **high variance** ($J_{cv} \gg J_{train}$), your model is too complex and is failing to generalize. Your goal is to **simplify the model** or provide it with more data to learn the general pattern.

- **1. Get more training examples:**
  - As seen in the learning curves, adding more data is one of the most effective ways to fix high variance. It makes it harder for a complex model to overfit the noise and forces it to learn the true underlying pattern.

- **2. Try a smaller set of features:**
  - Having too many features gives the model too much flexibility to overfit.
  - By reducing the number of features (feature selection), you reduce the model's complexity and its ability to fit a "wiggly" function to the training data.

- **3. Increase the regularization parameter ($\lambda$):**
  - A larger $\lambda$ increases the penalty on the parameters, forcing them to be smaller.
  - This results in a simpler, smoother function that is less prone to overfitting.

---

### **Fixing a High Bias Problem (Underfitting)**

If your diagnosis is **high bias** ($J_{train}$ is high), your model is too simple to even capture the patterns in the training data. Your goal is to **make the model more complex** or give it more information to work with.

- **1. Get additional (new) features:**
  - If your model is trying to predict a house price _only_ from its size, it will have high bias. It's missing crucial information.
  - Adding new, relevant features (like the number of bedrooms, age of the house, etc.) gives the model more information to make a good prediction.

- **2. Add polynomial features (e.g., $x^2, x^3, x \cdot y$):**
  - This is a form of feature engineering that increases model complexity.
  - It allows the model to fit a non-linear curve instead of just a straight line, which can significantly reduce bias.

- **3. Decrease the regularization parameter ($\lambda$):**
  - A smaller $\lambda$ _reduces_ the penalty on the parameters.
  - This gives the model more flexibility to fit a more complex, "wiggly" curve to the training data, thereby reducing bias.

> **Note:** You should **not** try to fix high bias by _reducing_ your training set size. While this would lower $J_{train}$ (as seen in the learning curves), it would make your model's generalization (CV) error worse.

---

### **Summary of Actions**

| Diagnosis                          | Your Goal                                            | What to Try Next                                                                                          |
| :--------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **High Variance**<br>(Overfitting) | Simplify the model or<br>get more data.              | 1. **Get more training examples.**<br>2. **Try a smaller set of features.**<br>3. **Increase $\lambda$.** |
| **High Bias**<br>(Underfitting)    | Make the model more<br>complex or get more features. | 1. **Get additional features.**<br>2. **Add polynomial features.**<br>3. **Decrease $\lambda$.**          |
