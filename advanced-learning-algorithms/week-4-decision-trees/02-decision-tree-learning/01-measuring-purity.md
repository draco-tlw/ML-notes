### **Entropy: Measuring Impurity 🪙**

In decision tree learning, the algorithm needs a way to measure the "purity" of a set of data.

- A **pure** set is one where all examples belong to the same class (e.g., 100% "Cat" or 100% "Not Cat").
- An **impure** set is a mix of classes.

**Entropy**, denoted as $H(p_1)$, is a mathematical function used to measure **impurity**.

---

### **Understanding Entropy with Examples**

Let **$p_1$** be the fraction of positive examples ($y=1$, e.g., "Cat") in a set.

- **Maximum Impurity (Entropy = 1):**
  - This occurs when the data is a perfect 50/50 split.
  - _Example:_ 3 Cats, 3 Dogs $\rightarrow p_1 = 0.5 \rightarrow H(p_1) = 1$.

- **No Impurity (Entropy = 0):**
  - This occurs when the data is 100% pure (all one class).
  - _Example 1:_ 6 Cats, 0 Dogs $\rightarrow p_1 = 1.0 \rightarrow H(p_1) = 0$.
  - _Example 2:_ 0 Cats, 6 Dogs $\rightarrow p_1 = 0.0 \rightarrow H(p_1) = 0$.

- **Some Impurity (0 < Entropy < 1):**
  - This occurs with any other mixed set.
  - _Example 1:_ 5 Cats, 1 Dog $\rightarrow p_1 \approx 0.83 \rightarrow H(p_1) \approx 0.65$.
  - _Example 2:_ 2 Cats, 4 Dogs $\rightarrow p_1 \approx 0.33 \rightarrow H(p_1) \approx 0.92$.
  - Note that the 2/4 split ($p_1=0.33$) is _more impure_ (closer to 0.5) than the 5/1 split ($p_1=0.83$), so its entropy is higher.

---

### **The Entropy Formula 🧮**

Let $p_1$ be the fraction of positive examples and $p_0 = 1 - p_1$ be the fraction of negative examples.

The formula for entropy is:
$$H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$$
or equivalently:
$$H(p_1) = -p_1 \log_2(p_1) - (1-p_1) \log_2(1-p_1)$$

**Key Conventions:**

- **Log Base 2 ($\log_2$):** This is used by convention so that the maximum impurity (at $p_1=0.5$) is exactly **1**.
- **Special Case ($0 \log_2 0$):** Since $\log_2(0)$ is undefined, we use the convention that **$0 \cdot \log_2(0) = 0$**. This ensures that pure sets ($p_1=0$ or $p_0=0$) correctly result in an entropy of 0.

---

### **Alternatives to Entropy**

- Other measures of impurity exist, such as the **Gini criteria** (or Gini impurity).
- The Gini function looks very similar to the entropy curve and serves the same purpose: to quantify the impurity of a node.
- This course will focus on using entropy.
