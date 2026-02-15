## **Feature Scaling Methods 📏**

There are several common techniques to scale features so they fall within a similar range, which helps gradient descent converge faster.

---

### **1. Max Scaling**

This is one of the simplest methods. You just divide each feature by its maximum value in the training set.

- **Formula:** $x_{scaled} = \frac{x_{original}}{max(x)}$
- **Resulting Range:** The scaled feature will typically range from just above 0 to 1.

---

### **2. Mean Normalization**

This method centers the feature values around zero.

- **Process:**
  1. Calculate the **mean** (average), $\mu$, of the feature in the training set.
  2. Calculate the **range** of the feature (i.e., $max(x) - min(x)$).
- **Formula:** $x_{scaled} = \frac{x_{original} - \mu}{max(x) - min(x)}$
- **Resulting Range:** The scaled feature will be centered around 0.

---

### **3. Z-score Normalization**

This is a very common technique that scales features based on their mean and standard deviation.

- **Process:**
  1. Calculate the **mean**, $\mu$, of the feature.
  2. Calculate the **standard deviation**, $\sigma$, of the feature.
- **Formula:** $x_{scaled} = \frac{x_{original} - \mu}{\sigma}$
- **Resulting Range:** The scaled feature will have a mean of 0 and a standard deviation of 1.

---

### **Rule of Thumb 🎯**

The general goal of feature scaling is to get your features into a range of roughly **-1 to +1**.

- **Acceptable Ranges:** Ranges like **-3 to +3** or **-0.3 to +0.3** are perfectly fine and don't necessarily need scaling.
- **Ranges to Scale:** You should consider scaling features with very large or very small ranges, as these can slow down gradient descent.
  - **Too Large:** A range like **-100 to +100**.
  - **Too Small:** A range like **-0.001 to +0.001**.

**When in doubt, it's almost never harmful to perform feature scaling.**
