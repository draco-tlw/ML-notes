## **The Anomaly Detection Algorithm 🔍**

We can now build the full algorithm using the concept of the Gaussian distribution.

### **The Model: Multiplying Probabilities**

We have a training set where each example $x$ has $n$ features ($x_1, \dots, x_n$).

- _Example:_ Aircraft engine with 2 features (Heat, Vibration). $n=2$.

To estimate the total probability $p(x)$ of an entire vector of features, we assume the features are statistically independent (though the algorithm works well even if they aren't) and **multiply the probabilities of each individual feature**.

$$p(x) = p(x_1; \mu_1, \sigma_1^2) \times p(x_2; \mu_2, \sigma_2^2) \times \dots \times p(x_n; \mu_n, \sigma_n^2)$$

This can be written compactly as:
$$p(x) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2)$$

- $\prod$ is the product symbol (like $\sum$ for summation), meaning "multiply all these terms together."
- Each feature $j$ has its own mean $\mu_j$ and variance $\sigma_j^2$.

---

### **The Algorithm Steps**

#### **1. Choose Features**

Select features $x_i$ that you believe are indicative of anomalous behavior.

#### **2. Fit Parameters ($\mu$ and $\sigma^2$)**

Calculate the mean and variance for **each feature** $j$ using the training set.

- $\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}$ (Average of feature $j$)
- $\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2$ (Variance of feature $j$)

#### **3. Compute $p(x)$ for New Examples**

Given a new example $x$ (e.g., a new engine), calculate its total probability:

$$p(x) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)$$

#### **4. Evaluation (Flag Anomaly)**

Compare the calculated $p(x)$ to a threshold $\epsilon$.

- **If $p(x) < \epsilon$:** Flag as **ANOMALY**.
- **If $p(x) \ge \epsilon$:** Consider it **NORMAL**.

---

### **Intuition**

- The algorithm fits a bell curve to _each_ feature independently.
- If a new example has even **one** feature that is very far from the mean (e.g., excessively high heat), the probability $p(x_j)$ for that specific feature will be tiny.
- Because we are multiplying, **multiplying by a tiny number makes the total product tiny.**
- Therefore, if any single feature is anomalous, the total $p(x)$ will drop below $\epsilon$, and the system will flag it.

---

### **Example Visualization**

- Imagine data with 2 features.
  - $x_1$: Heat (Mean=5, Sigma=2)
  - $x_2$: Vibration (Mean=3, Sigma=1)
- Multiplying these two distributions creates a 3D "hill" of probability.
  - The peak (highest probability) is in the center, where the means are.
  - The edges (low probability) are where anomalies live.
- **Test 1 (Center):** $p(x) = 0.4$ (Normal)
- **Test 2 (Far Edge):** $p(x) = 0.0021$ (Anomaly)
