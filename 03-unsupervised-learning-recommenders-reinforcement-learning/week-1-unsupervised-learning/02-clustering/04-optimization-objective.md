## **The K-Means Cost Function 📉**

K-means is not just moving points around randomly; it is mathematically optimizing a specific cost function.

### **The Objective (Minimizing Distortion)**

The cost function $J$ (also called the **Distortion Function**) measures the sum of squared distances between every training example and its assigned cluster centroid.

$$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_K) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$$

- $x^{(i)}$: The $i$-th training example.
- $c^{(i)}$: The index of the cluster to which $x^{(i)}$ is currently assigned.
- $\mu_{c^{(i)}}$: The location of the centroid for that cluster.
- $||...||^2$: The squared Euclidean distance.

**Goal:** The algorithm tries to find the assignments ($c$) and the centroid locations ($\mu$) that minimize this cost function $J$.

---

### **How the Algorithm Minimizes $J$**

The two steps of the K-means algorithm correspond exactly to minimizing this function with respect to different variables:

1.  **Step 1 (Assign Points):**
    - While holding the centroids ($\mu$) fixed, we choose the assignments ($c^{(i)}$) that minimize the distance squared.
    - Assigning a point to the _closest_ centroid is mathematically guaranteed to minimize this part of the cost function.

2.  **Step 2 (Move Centroids):**
    - While holding the assignments ($c$) fixed, we choose the centroid locations ($\mu$) that minimize the cost.
    - Setting the centroid to be the **mean (average)** of the points assigned to it is mathematically guaranteed to minimize the sum of squared distances for those points.

---

### **Convergence Guarantee**

- Because both steps of the algorithm are designed to reduce (or maintain) the cost function $J$, the K-means algorithm is **guaranteed to converge**.
- The cost function will decrease with every iteration until it settles at a minimum.
- **Debugging Tip:** If you plot the cost function over iterations and see it _increasing_, there is a bug in your code.
- **Stopping Criteria:** You can declare convergence when the cost function stops decreasing (or decreases very slowly).
