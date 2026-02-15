## **Collaborative Filtering for Binary Labels**

In many real-world applications, user feedback isn't a 1-5 star rating. It's often binary: a simple "Yes/No," "Clicked/Didn't Click," or "Watched/Didn't Watch."

### **The Data**

The labels in our dataset change from ratings (0-5) to:

- **1:** The user engaged (Liked, Purchased, Watched > 30s, Clicked).
- **0:** The user did _not_ engage (Did not buy, skipped the video, ignored the ad) after being shown the item.
- **? (Question Mark):** The user has not yet seen the item.

### **The Transformation: From Linear to Logistic**

Just as we moved from Linear Regression to Logistic Regression in supervised learning, we do the same here.

- **Old Prediction (Ratings):**
  $$f(x) = w^{(j)} \cdot x^{(i)} + b^{(j)}$$
  (This output a continuous number like 4.5).

- **New Prediction (Binary Probability):**
  We wrap the linear equation in the **sigmoid (logistic) function** $g(z)$ to force the output between 0 and 1.
  $$g(z) = \frac{1}{1 + e^{-z}}$$
  $$f(x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})$$
  This output represents the **probability** that user $j$ will engage with item $i$ ($y=1$).

---

### **The Cost Function (Binary Cross-Entropy)**

Since we are now doing classification (0 or 1), the "Mean Squared Error" cost function is no longer appropriate. We switch to the **Binary Cross-Entropy Loss** (the standard loss for logistic regression).

**Loss for a single example:**
$$Loss(f(x), y) = -y \log(f(x)) - (1-y) \log(1-f(x))$$

**Total Cost Function:**
We sum this loss over all pairs $(i, j)$ where a label exists ($r(i,j)=1$).

$$J(w, b, x) = \sum_{(i,j):r(i,j)=1} \text{Loss}(f_{w,b,x}(x), y^{(i,j)}) + \text{Regularization}$$

By minimizing this cost function using gradient descent, the algorithm learns to predict the **probability** of a user liking an item.

---

### **Summary**

- **Input:** Binary data (1 for engaged, 0 for not engaged).
- **Model:** Logistic function applied to the dot product of user/item vectors.
- **Optimization:** Minimize binary cross-entropy loss.
- **Result:** A system that predicts the _likelihood_ of engagement, which is perfect for recommending items (e.g., "Show the user the items with the highest predicted probability of a click").
