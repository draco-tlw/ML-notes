### 1. Preprocessing: Normalization

Before the algorithm starts, you must normalize the data.

- **Zero Mean:** You subtract the mean from each feature so the data is centered at the origin (0,0).
- **Feature Scaling:** If $x_1$ is a huge number (like house size in square feet) and $x_2$ is a small number (like number of bedrooms), one feature will dominate the other. You scale them so they have a similar range.

### 2. Finding the New Axis (The Principal Component)

The goal is to find a new axis (let's call it the **z-axis**) that best captures the information in the data.

- **Projection:** PCA tests different directions for this new axis. For every data point, it draws a line perpendicular (at 90 degrees) to the new axis. This is called **projecting** the data.
- **Maximizing Variance:** How does it choose the best axis? It looks for the direction that keeps the data points **spread out** as much as possible.
  - **Bad Choice:** If the projected points are squished together, you have lost information (low variance).
  - **Good Choice:** If the projected points are widely spread out, you have retained the maximum amount of information (high variance).

This best axis is called the **Principal Component**.

### 3. The Math: Dot Products

Once PCA finds the direction of the z-axis, it defines it using a "unit vector" (a vector of length 1). Let's say this vector is $[0.71, 0.71]$.

To convert a data point (e.g., $x_1=2, x_2=3$) into the new feature $z$:
$$z = \text{Vector}_{data} \cdot \text{Vector}_{axis}$$
$$z = [2, 3] \cdot [0.71, 0.71] = 3.55$$

Your new feature $z$ is **3.55**. You have successfully reduced 2 dimensions into 1.

### 4. What if you want more dimensions?

If you have 50 features and want to reduce them to 2 (instead of 1), PCA finds the second axis ($z_2$) by looking for a direction that is:

1.  **Perpendicular (90 degrees)** to the first axis ($z_1$).
2.  Captures the **next highest** amount of variance.

### 5. PCA is NOT Linear Regression

It is easy to confuse the two, but they do completely different things:

| Feature               | Linear Regression                                                      | PCA                                                                    |
| :-------------------- | :--------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| **Type**              | Supervised Learning (uses labels $y$)                                  | Unsupervised Learning (no labels)                                      |
| **Goal**              | Predict $y$ given $x$                                                  | Reduce features / Visualization                                        |
| **Error Calculation** | Minimizes **Vertical** distance (squared error between point and line) | Minimizes **Perpendicular** distance (distance from point to the axis) |

### 6. Reconstruction

You can also go backward. If you have the value $z=3.55$, you can try to recreate the original $x_1$ and $x_2$.

- **Formula:** $z \times \text{Vector}_{axis}$
- **Result:** $3.55 \times [0.71, 0.71] = [2.52, 2.52]$
