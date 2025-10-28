## **Matrix Multiplication (Optional)**

This section explains how matrix multiplication works, building up from the simpler vector dot product. This is the core mathematical operation that makes vectorization possible.

### **1. Vector-Vector Dot Product**

The **dot product** is a fundamental operation where you take two vectors of the same size, multiply their corresponding elements, and sum the results.

- **Example:**
  - $\vec{a} = [1, 2]$
  - $\vec{w} = [3, 4]$
  - **Dot Product:** $z = (1 \times 3) + (2 \times 4) = 3 + 8 = 11$
- **General Formula:** $z = a_1w_1 + a_2w_2 + \dots + a_nw_n$

---

### **2. The Transpose and the "Row $\times$ Column" Rule**

- A **column vector** is a matrix with one column (e.g., a 2x1 matrix).
- A **row vector** is a matrix with one row (e.g., a 1x2 matrix).
- The **transpose** operation (denoted by $A^T$) converts a column vector into a row vector (it lays the vector "on its side").
- **Key Equivalence:** The vector dot product $\vec{a} \cdot \vec{w}$ is mathematically equivalent to the matrix multiplication of $\vec{a}^T$ (a row vector) and $\vec{w}$ (a column vector).
  - $z = \vec{a} \cdot \vec{w} \equiv z = \vec{a}^T \times \vec{w}$

---

### **3. Vector-Matrix Multiplication**

This is the process of multiplying a row vector by a matrix.

- **Rule:** You perform a dot product between the **row vector** and **each column** of the matrix. The results form a new row vector.
- **Example:**
  - $\vec{a}^T = [1, 2]$ (a 1x2 matrix)
  - $W = \begin{bmatrix} 3 & 5 \\ 4 & 6 \end{bmatrix}$ (a 2x2 matrix, with columns $\begin{bmatrix} 3 \\ 4 \end{bmatrix}$ and $\begin{bmatrix} 5 \\ 6 \end{bmatrix}$)
- **Calculation:**
  1.  **First element of output:** $(\vec{a}^T) \cdot (\text{Col 1 of } W)$
      - $[1, 2] \cdot [3, 4] = (1 \times 3) + (2 \times 4) = 11$
  2.  **Second element of output:** $(\vec{a}^T) \cdot (\text{Col 2 of } W)$
      - $[1, 2] \cdot [5, 6] = (1 \times 5) + (2 \times 6) = 17$
- **Result:** $Z = [11, 17]$ (a 1x2 matrix)

---

### **4. Matrix-Matrix Multiplication**

This is the general form. It simply repeats the vector-matrix multiplication process for **every row** of the first matrix.

- **Rule:** The element at `(row i, col j)` of the output matrix is the dot product of `(row i)` of the _first_ matrix and `(col j)` of the _second_ matrix.

- **Example:** $Z = A^T \times W$
  - $A^T = \begin{bmatrix} 1 & 2 \\ -1 & -2 \end{bmatrix}$
  - $W = \begin{bmatrix} 3 & 5 \\ 4 & 6 \end{bmatrix}$

- **Calculation:**
  1.  **Row 1 of Z:** (Row 1 of $A^T$) $\times$ W
      - $([1, 2] \cdot [3, 4]) = 11$
      - $([1, 2] \cdot [5, 6]) = 17$
      - $\rightarrow$ First row of Z is $[11, 17]$
  2.  **Row 2 of Z:** (Row 2 of $A^T$) $\times$ W
      - $([-1, -2] \cdot [3, 4]) = (-3) + (-8) = -11$
      - $([-1, -2] \cdot [5, 6]) = (-5) + (-12) = -17$
      - $\rightarrow$ Second row of Z is $[-11, -17]$

- **Final Result:** $Z = \begin{bmatrix} 11 & 17 \\ -11 & -17 \end{bmatrix}$
