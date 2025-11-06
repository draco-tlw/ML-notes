## **General Matrix Multiplication (Optional)**

This section provides the general rule for matrix multiplication, which is the core operation for vectorizing a neural network.

### **The Transpose Operation ($A^T$)**

- A **matrix** is a 2D array of numbers, defined by its `(rows, columns)`.
- The **transpose** of a matrix, $A^T$, is found by taking the **columns** of $A$ and turning them into the **rows** of $A^T$.
- **Example:** If $A$ is a **2x3** matrix (2 rows, 3 cols), then $A^T$ will be a **3x2** matrix (3 rows, 2 cols).

---

### **The Rule for Matrix Multiplication: $Z = A \times W$ 🧮**

The core rule for multiplying two matrices (e.g., $A^T$ and $W$) is to compute the dot product of rows from the first matrix with columns from the second.

- To find the element in **row $i$** and **column $j$** of the output matrix $Z$, you take the dot product of **row $i$ of the first matrix** and **column $j$ of the second matrix**.
- **$Z(i, j) = (\text{Row } i \text{ of } A^T) \cdot (\text{Column } j \text{ of } W)$**

---

### **Example Calculation**

Let's calculate a few elements of $Z = A^T \times W$.

- **To find $Z(1, 1)$ (Row 1, Col 1):**
  - $(\text{Row 1 of } A^T) \cdot (\text{Col 1 of } W)$
  - $[1, 2] \cdot [3, 4] = (1 \times 3) + (2 \times 4) = \mathbf{11}$

- **To find $Z(3, 2)$ (Row 3, Col 2):**
  - $(\text{Row 3 of } A^T) \cdot (\text{Col 2 of } W)$
  - $[0.1, 0.2] \cdot [5, 6] = (0.1 \times 5) + (0.2 \times 6) = 0.5 + 1.2 = \mathbf{1.7}$

- **To find $Z(2, 3)$ (Row 2, Col 3):**
  - $(\text{Row 2 of } A^T) \cdot (\text{Col 3 of } W)$
  - $[-1, -2] \cdot [7, 8] = (-1 \times 7) + (-2 \times 8) = -7 - 16 = \mathbf{-23}$

This process is repeated for every element to compute the full output matrix $Z$.

---

### **Dimension Rules for Multiplication 📏**

You cannot multiply matrices of any arbitrary size. They must follow two specific rules.

- **1. Requirement:** To multiply `Matrix1` $\times$ `Matrix2`, the number of **columns** in `Matrix1` must **equal** the number of **rows** in `Matrix2`.
  - _Example:_ $A^T$ (a **3x2** matrix) $\times$ $W$ (a **2x4** matrix).
  - The "inner dimensions" (2 and 2) match, so the operation is valid.
  - This is required because the dot product must be between vectors of the same length.

- **2. Output Shape:** The resulting matrix, $Z$, will have the "outer dimensions."
  - _Shape:_ (Number of **rows** from `Matrix1`) $\times$ (Number of **columns** from `Matrix2`).
  - _Example:_ $Z$ will be a **3x4** matrix.

---

### **Key Takeaway**

This matrix multiplication rule is the key to **vectorization**. It allows a neural network to perform all the dot products for an entire layer in one highly efficient operation, which is why `np.matmul` runs so much faster than a `for` loop.
