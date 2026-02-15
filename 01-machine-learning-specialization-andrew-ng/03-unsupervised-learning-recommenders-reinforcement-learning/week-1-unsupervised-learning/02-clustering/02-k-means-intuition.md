## **How K-Means Clustering Works**

K-means is an iterative algorithm that automatically groups unlabeled data into clusters. It works by repeatedly performing two specific steps until it finds the best grouping.

### **The Algorithm**

1.  **Initialization:**
    - The algorithm starts by taking a **random guess** at where the centers of the clusters might be.
    - It picks $K$ random points (where $K$ is the number of clusters you want to find). These points are called **Cluster Centroids**.

2.  **The Iterative Loop:**
    K-means repeatedly cycles through two main steps:
    - **Step 1: Assign points to cluster centroids**
      - The algorithm goes through every data point ($x^{(1)}$ to $x^{(m)}$).
      - For each point, it calculates the distance to the centroids.
      - It assigns (or "colors") the point to the cluster centroid it is **closest to**.

    - **Step 2: Move cluster centroids**
      - The algorithm looks at all the points assigned to a specific cluster (e.g., all the red points).
      - It calculates the **average (mean)** location of those points.
      - It moves the centroid to that new average location.

### **Convergence**

- After moving the centroids (Step 2), the algorithm goes back to Step 1 and re-assigns points based on the _new_ centroid locations. Some points might change clusters (change color) because the centroids have moved.
- This cycle (Assign $\rightarrow$ Move $\rightarrow$ Assign $\rightarrow$ Move) repeats.
- **Convergence** occurs when there are no more changes:
  - Points stop switching clusters (colors remain the same).
  - Centroids stop moving (their location is stable).

At this point, the algorithm has successfully grouped the data into distinct clusters.
