### **The K-Means Algorithm**

The algorithm takes a dataset $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$ and the number of clusters $K$ as input.

#### **Step 1: Random Initialization**

Randomly initialize $K$ cluster centroids: $\mu_1, \mu_2, ..., \mu_K$.

- These centroids are vectors with the same dimension ($n$) as your training examples.
- _Example:_ If your data has 2 features (like height and weight), each centroid $\mu$ will also be a vector of 2 numbers.

#### **Step 2: The Iterative Loop**

The algorithm repeatedly performs these two steps until convergence:

**A. Assign Points to Cluster Centroids**
For every training example $x^{(i)}$ (from $i=1$ to $m$), determine which centroid is closest.

- We calculate the distance using the standard Euclidean distance (L2 norm): $||x^{(i)} - \mu_k||$.
- _Note:_ In code, it is computationally convenient to minimize the **squared** distance: $||x^{(i)} - \mu_k||^2$.
- We set a variable $c^{(i)}$ to be the index ($1$ to $K$) of the closest cluster centroid.

$$c^{(i)} := \text{index } k \text{ that minimizes } ||x^{(i)} - \mu_k||^2$$

**B. Move Cluster Centroids**
For every cluster $k$ (from $1$ to $K$), update the centroid to be the **average (mean)** of all points assigned to that cluster.

- _Example:_ If cluster 1 has assigned points $x^{(1)}, x^{(5)}, x^{(6)}, x^{(10)}$, the new centroid location is:
  $$\mu_1 = \frac{1}{4} (x^{(1)} + x^{(5)} + x^{(6)} + x^{(10)})$$

---

### **Corner Case: Empty Clusters**

What happens if a cluster has **zero** points assigned to it during the assignment step? (e.g., trying to average 0 points).

1.  **Eliminate the cluster (Most Common):** simply remove that centroid. You will end up with $K-1$ clusters.
2.  **Reinitialize:** If you strictly require $K$ clusters, you can randomly re-initialize that specific centroid and run the loop again.

---

### **Application: Non-Separated Data**

K-means is often used on data that looks like a single "blob" rather than distinct, separated islands.

- **Example: T-Shirt Sizing**
  - **Data:** Heights and Weights of customers (continuous distribution).
  - **Goal:** Determine optimal sizing for Small, Medium, and Large.
  - **Process:** Run K-means with $K=3$.
  - **Result:** The algorithm groups the data into three clusters. The centroids ($\mu_1, \mu_2, \mu_3$) represent the "prototypical" measurements for the three sizes, ensuring the shirts fit the largest number of people effectively.
