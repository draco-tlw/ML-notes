# Note: Vectorization and Python Implementation

**Source:** Coursera - Neural Networks and Deep Learning (Week 2)

## 1. Vectorization

**Definition:** Vectorization is the art of eliminating explicit `for` loops in code.

- **Importance:** In the deep learning era, training occurs on large datasets. Algorithms must run quickly to avoid excessive wait times.

- **Performance:** Vectorized code takes advantage of **SIMD** (Single Instruction Multiple Data) parallelization instructions available in both CPUs and GPUs.

- _Example:_ In a demo comparing a dot product (w^T x) implementation, the vectorized version was approximately **300 times faster** than the explicit `for` loop version (1.5ms vs. ~480ms).

- **Rule of Thumb:** Whenever possible, avoid using explicit `for` loops.

### Examples of Vectorization

- **Dot Product:** Instead of looping i = 1 \dots n, use `z = np.dot(w, x)`.

- **Matrix-Vector Multiplication:** To compute u = Av, use `u = np.dot(A, v)` instead of nested loops over i and j.

- **Element-wise Operations:** Use NumPy built-in functions to apply operations to entire vectors at once:
- `np.exp(v)`

- `np.log(v)`

- `np.abs(v)`

- `v**2`

## 2. Vectorizing Logistic Regression

We can vectorize both the forward and backward propagation steps to process the entire training set (m examples) without a loop over training examples.

### Forward Propagation

- **Input:** Matrix X of shape (n_x, m) (inputs stacked in columns).

- **Computation:**

- **Python Implementation:**
- `Z = np.dot(w.T, X) + b`.

- _Note:_ Python automatically "broadcasts" the real number b to a 1 \times m row vector to match the shape of w^T X.

### Backward Propagation (Gradient Computation)

- **Input:** A (predictions) and Y (labels), both stacked horizontally.

- **Computation:**
-

.

-

\rightarrow `db = (1/m) * np.sum(dZ)`.

-

\rightarrow `dw = (1/m) * np.dot(X, dZ.T)`.

- **Result:** You can implement a single iteration of Gradient Descent (Forward Prop + Back Prop + Parameter Update) without any explicit `for` loops. An outer loop is still required for the number of training iterations (epochs).

## 3. Broadcasting in Python

Broadcasting allows Python to perform operations between arrays of different shapes by implicitly copying the smaller array to match the larger one.

**General Principle:**
If you perform an operation between an (m, n) matrix and a (1, n) matrix (or vector), Python copies the (1, n) matrix m times vertically to create an (m, n) matrix, then performs the operation element-wise.

**Examples:**

- **Matrix + Scalar:** Adding 100 to a vector adds 100 to _every_ element.

- **Matrix + Row Vector:** Adding a (1, n) vector to an (m, n) matrix adds the vector to every row.

- **Matrix + Column Vector:** Adding an (m, 1) vector to an (m, n) matrix adds the vector to every column.

## 4. Python/NumPy Tips and Best Practices

Python's flexibility can lead to subtle bugs, particularly regarding vector shapes.

- **Rank 1 Arrays:** Avoid data structures with shape `(5,)`. These are neither column nor row vectors and behave inconsistently.

- _Issue:_ Transposing a rank 1 array `a.T` looks identical to `a`.

- **Recommendation:** Always commit to a specific dimension.
- Use `(5, 1)` for column vectors.

- Use `(1, 5)` for row vectors.

- **Debugging Tools:**
- **Reshape:** Use `a = a.reshape((5, 1))` to ensure dimensions are correct. This is a cheap O(1) operation.

- **Assert:** Use `assert(a.shape == (5, 1))` to document code and catch dimension mismatch errors early.

## 5. Justification for Logistic Regression Cost Function

**Source:** Optional Video

- **Probabilistic Interpretation:**
- If y=1, P(y|x) = \hat{y}.

- If y=0, P(y|x) = 1 - \hat{y}.

- Combined equation: P(y|x) = \hat{y}^y (1-\hat{y})^{(1-y)}.

- **Log Likelihood:**
- Taking the log: \log P(y|x) = y \log \hat{y} + (1-y) \log (1-\hat{y}).

- This formula matches the negative of the Loss function.

- **Maximum Likelihood Estimation (MLE):**
- To find the best parameters, we want to **maximize** the probability (likelihood) of the labels in the training set.

- Maximizing the log-likelihood is equivalent to **minimizing** the Cost Function J (which is the negative sum of logs).

- This provides the statistical justification for using the specific cost function in Logistic Regression.
