## **Content-Based Filtering: Using Features**

In this approach, we assume we _have_ specific features for each item (movie) and we use these features to predict a user's preference.

### **The Setup**

- **$n_u$**: Number of users.
- **$n_m$**: Number of movies (items).
- **$n$**: Number of features per movie.
  - _Example:_ $n=2$, where $x_1$ = Romance score, $x_2$ = Action score.

We treat this like a **Linear Regression** problem, but with a twist: we fit a separate linear regression model for **each individual user**.

### **The Model**

For every user $j$, we learn a set of parameters $w^{(j)}$ and $b^{(j)}$.
To predict the rating user $j$ would give to movie $i$:
$$\text{Prediction} = w^{(j)} \cdot x^{(i)} + b^{(j)}$$

- **$x^{(i)}$**: The feature vector for movie $i$ (e.g., [0.99, 0] for a romance movie).
- **$w^{(j)}$**: The user's "preference" vector (e.g., [5, 0] if they love romance and hate action).
- **$b^{(j)}$**: The user's bias term.

---

### **The Cost Function (For One User $j$)**

We want to minimize the squared error between the predicted rating and the actual rating, summing only over movies the user has actually rated ($r(i,j)=1$).

$$J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i:r(i,j)=1} \left( (w^{(j)} \cdot x^{(i)} + b^{(j)}) - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2$$

- The first term is the **Mean Squared Error** (prediction vs. actual).
- The second term is the **Regularization** term (to prevent overfitting).
- _Note:_ The normalization constant $1/m^{(j)}$ is removed for mathematical convenience, as it doesn't change the optimal parameters.

---

### **The Full Cost Function (For All Users)**

To learn the preferences for _all_ users simultaneously, we sum the cost function over all users ($j=1$ to $n_u$).

$$J(\text{all } w, \text{all } b) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left( (w^{(j)} \cdot x^{(i)} + b^{(j)}) - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2$$

By minimizing this total cost function using **Gradient Descent**, we can learn the optimal parameters ($w, b$) for every single user, allowing us to predict their ratings for movies they haven't seen yet.

---

### **The Problem**

This method works great **IF** we have the features $x^{(i)}$ (Romance score, Action score, etc.). But in the real world, we rarely have this detailed data for every single movie.
