## **Unsupervised Learning: Formal Definition & Other Types**

### **Formal Definition**

- **Supervised Learning:** The data provided is a set of **(x, y)** pairs, where `x` is the input and `y` is the correct output label.
- **Unsupervised Learning:** The data provided is only a set of inputs **(x)**, with **no corresponding output labels**. The goal is for the algorithm to discover interesting structures or patterns within this unlabeled data.

---

### **Types of Unsupervised Learning**

Beyond clustering, there are other important types of unsupervised learning algorithms.

1. **Clustering (Recap):**

   - Groups similar data points together.
   - _Example:_ Grouping news articles or segmenting customers.

2. **Anomaly Detection:**

   - Used to identify rare or unusual data points/events.
   - _Example:_ **Fraud detection**. An algorithm can analyze many financial transactions to learn what a "normal" transaction looks like and then flag unusual ones that might be fraudulent.

3. **Dimensionality Reduction:**
   - Compresses a large dataset into a much smaller one while losing as little information as possible.
   - This is useful for data visualization, saving storage space, and speeding up other learning algorithms.

---

### **Reinforcing the Concepts (Quiz Summary)**

Here's how to categorize different problems:

- **Spam Filtering (Supervised):** Requires a dataset of emails already labeled as "spam" or "not spam."
- **Grouping News Articles (Unsupervised):** Uses a clustering algorithm to find related articles without any pre-existing labels.
- **Market Segmentation (Unsupervised):** Uses a clustering algorithm to discover customer groups automatically from unlabeled data.
- **Diabetes Diagnosis (Supervised):** Requires a dataset of patient records labeled as "diabetes" or "no diabetes," similar to the breast cancer classification example.
