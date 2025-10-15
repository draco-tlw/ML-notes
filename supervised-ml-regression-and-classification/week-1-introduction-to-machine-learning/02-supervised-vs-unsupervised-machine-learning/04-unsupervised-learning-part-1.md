## **Unsupervised Learning**

**Unsupervised learning** is the second major type of machine learning. It is used when the data provided to the algorithm has **no output labels** (`y`).

### **How it Works 🧠**

- Unlike supervised learning, where the goal is to map an input `x` to a known output `y`, unsupervised learning is given a dataset consisting only of inputs.
- The algorithm's task is to find **structure, patterns, or something interesting** within the data on its own, without any "right answers" to guide it.
- **Example:** Given data on patients' tumor sizes and ages, but _without_ labels indicating if the tumors are benign or malignant, an unsupervised algorithm might identify that the data points naturally fall into two distinct groups.

---

### **Clustering: A Key Type of Unsupervised Learning**

One of the most common types of unsupervised learning is **clustering**.

- **Goal:** A clustering algorithm takes unlabeled data and automatically groups it into different clusters based on similarity.

---

### **Examples of Clustering in Action**

1. **Google News:**

   - Every day, the algorithm analyzes hundreds of thousands of news articles.
   - It identifies articles with similar words and topics (e.g., "panda," "twins," "zoo") and groups them together into a single news story.
   - It does this **without human supervision**, automatically discovering the day's emerging topics.

2. **Genetics (DNA Analysis):**

   - Given genetic data from many individuals, a clustering algorithm can group people into different categories based on their gene activity patterns.
   - This can help researchers discover different genetic profiles or types of individuals within a population without knowing those types in advance.

3. **Market Segmentation:**
   - Companies use clustering to analyze their customer databases.
   - The algorithm groups customers into different market segments based on their behaviors, preferences, or motivations.
   - This allows the company to better understand and serve different types of customers. For example, an education company might find distinct clusters of learners motivated by career growth vs. pure knowledge acquisition.
