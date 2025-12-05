## **The Gaussian (Normal) Distribution 🔔**

To perform anomaly detection, we need a way to calculate the probability $p(x)$ that a specific data point belongs to a dataset. We do this using the **Gaussian Distribution**, also known as the **Normal Distribution** or the **Bell Curve**.

### **Key Parameters**

The shape of the Gaussian distribution is determined by two parameters:

1.  **$\mu$ (Mu) - The Mean:**
    - This describes the **center** of the distribution.
    - It determines where the peak of the bell curve sits.
2.  **$\sigma$ (Sigma) - The Standard Deviation:**
    - This describes the **width** of the distribution.
    - It determines how spread out the data points are.
    - **$\sigma^2$ (Sigma squared)** is called the **Variance**.

### **Visualizing the Distribution**

The area under the curve always sums to **1** (representing total probability). Therefore, changing the width ($\sigma$) forces the height to change as well.

| Scenario                                | Shape of Curve      | Interpretation                                  |
| :-------------------------------------- | :------------------ | :---------------------------------------------- |
| **Standard** ($\mu=0, \sigma=1$)        | Standard Bell Shape | Centered at 0, standard spread.                 |
| **Small $\sigma$** (e.g., $\sigma=0.5$) | **Tall and Skinny** | Data is tightly clustered around the mean.      |
| **Large $\sigma$** (e.g., $\sigma=2$)   | **Short and Wide**  | Data is very spread out (high variance).        |
| **Change $\mu$**                        | Shifts Left/Right   | The shape stays the same, but the center moves. |

---

### **The Mathematical Formula**

The probability of $x$, denoted as $p(x)$, is calculated using the following formula:

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- This formula produces the bell-shaped curve.
- Points near the mean $\mu$ will produce a high value for $p(x)$.
- Points far from the mean (in the "tails") will produce a very low value for $p(x)$.

---

### **Parameter Estimation: Learning from Data**

If you have a dataset of $m$ examples ($x^{(1)}, \dots, x^{(m)}$), how do you find the correct $\mu$ and $\sigma$ to fit your data?

You calculate the **Maximum Likelihood Estimates**:

1.  **Estimate the Mean ($\mu$):**
    Simply calculate the average of your training examples.
    $$\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$

2.  **Estimate the Variance ($\sigma^2$):**
    Calculate the average of the squared differences between each example and the mean.
    $$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2$$

> **Note on Statistics:** Some statistics courses recommend dividing by $m-1$ instead of $m$ for the variance. In machine learning (and for large datasets), the difference is negligible. Using $1/m$ is standard practice here.

---

### **Applying to Anomaly Detection**

Once you have calculated $\mu$ and $\sigma^2$ for your data:

- **Normal examples:** Will fall near the center ($\mu$). $p(x)$ will be **high**.
- **Anomalous examples:** Will fall far away in the tails. $p(x)$ will be **low**.

Currently, we have looked at $x$ as a single number (one feature). However, real-world problems usually involve multiple features ($x_1, x_2, \dots, x_n$).
