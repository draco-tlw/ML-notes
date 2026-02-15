# Note: Neural Network Basics & Logistic Regression

**Source:** Coursera - Neural Networks and Deep Learning (Week 2)

## 1. Binary Classification

To build a neural network, we first need to understand how to process input data for binary classification (e.g., Is this image a cat (1) or not (0)?).

- **Image Representation:** Computers store images as three separate matrices corresponding to Red, Green, and Blue (RGB) color channels.

- **Feature Vector (x):** To input an image into an algorithm, we "unroll" these pixel intensity values into a single feature vector.

- If an image is 64 \times 64 pixels, the dimension of x is 64 \times 64 \times 3 = 12,288.

- We denote this input dimension as n_x (or simply n).

### Key Notation

- **Training Example:** represented as a pair (x, y), where x \in \mathbb{R}^{n_x} and y \in \{0, 1\}.

- **Training Set Size:** m denotes the number of training examples.

- **Input Matrix (X):**
- Created by stacking inputs x^{(1)}, x^{(2)}, ..., x^{(m)} in **columns**.

- Dimensions: X is an n_x \times m matrix.

- _Note:_ This column-stacking convention makes implementation easier compared to row-stacking.

- **Output Matrix (Y):**
- Created by stacking labels y^{(1)}, y^{(2)}, ..., y^{(m)} in columns.

- Dimensions: Y is a 1 \times m matrix.

## 2. Logistic Regression Model

Logistic Regression is a learning algorithm used for binary classification (outputting 0 or 1).

- **Goal:** We want to output a prediction \hat{y} which estimates the probability that y=1 given input x.

- \hat{y} = P(y=1 | x).

- Output must be between 0 and 1.

- **Linear Function Issue:** A standard linear function w^T x + b can be much larger than 1 or even negative, which is invalid for probability.

- **Sigmoid Function:** We apply the sigmoid function to the linear output to constrain it between 0 and 1.

- **Formula:** \hat{y} = \sigma(z), where z = w^T x + b.

- **Sigmoid Definition:** \sigma(z) = \frac{1}{1 + e^{-z}}.

- **Behavior:** If z is large, \sigma(z) \approx 1. If z is a large negative number, \sigma(z) \approx 0.

- **Parameters:**
- w: An n_x-dimensional vector.

- b: A real number (intercept/bias).

## 3. Cost Function

To train parameters w and b, we need to measure how well the model predicts.

### Loss Function (Single Example)

- **Squared Error:** Commonly used in linear regression (\frac{1}{2}(\hat{y}-y)^2), but typically **not** used in logistic regression because it leads to a non-convex optimization problem (multiple local optima).

- **Logistic Loss Function:** We use a loss function that results in a convex problem (single global optimum).

- **Formula:** L(\hat{y}, y) = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y})).

- **Intuition:**
- If y=1: Loss is -\log(\hat{y}). To minimize loss, \hat{y} must be large (close to 1).

- If y=0: Loss is -\log(1-\hat{y}). To minimize loss, \hat{y} must be small (close to 0).

### Cost Function (Entire Training Set)

The Cost Function J is the average of the Loss Functions across the entire training set.

- **Formula:** J(w, b) = \frac{1}{m} \sum\_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)}).

- **Goal:** Find w, b that minimize J(w, b).

## 4. Gradient Descent

Gradient descent is the algorithm used to minimize the cost function J.

- **Convexity:** Because the logistic cost function is convex (bowl-shaped), gradient descent will converge to the global optimum regardless of initialization.

- **The Algorithm:**
  Repeat until convergence:
- w := w - \alpha \frac{dJ(w,b)}{dw}.

- b := b - \alpha \frac{dJ(w,b)}{db}.

- Here, \alpha is the **learning rate** (controls step size).

- The derivative term represents the **slope** (direction of steepest descent).

## 5. Computation Graph & Derivatives

Neural network computations are organized into a **Forward Pass** (output computation) and a **Backward Pass** (derivative computation).

- **Derivatives Intuition:** A derivative is simply the slope of a function at a specific point. It measures how much the output changes if you nudge the input slightly.

- **Computation Graph:** Visualizes the flow of calculation.
- **Forward (Left-to-Right):** Computes the cost J. (e.g., x, w, b \rightarrow z \rightarrow a \rightarrow L) .

- **Backward (Right-to-Left):** Computes derivatives using the **Chain Rule**.

- **Chain Rule Example:** If J depends on v, and v depends on a, then \frac{dJ}{da} = \frac{dJ}{dv} \cdot \frac{dv}{da}.

### Logistic Regression Derivatives

To implement gradient descent, we compute derivatives backward from the Loss L:

1.  **da (Derivative of Loss wrt a):** \frac{dL}{da} = -\frac{y}{a} + \frac{1-y}{1-a}.

2.  **dz (Derivative of Loss wrt z):** dz = a - y.

- _Note:_ This is a simplification derived from the chain rule: \frac{dL}{dz} = \frac{dL}{da} \cdot \frac{da}{dz}.

3. **dw and db (derivatives wrt parameters):**

- dw_1 = x_1 \cdot dz.

- db = dz.

## 6. Implementation Notes

- **Accumulators:** When processing m examples, we sum the derivatives (dw, db) for each example and then divide by m to get the average.

- **Vectorization Teaser:** Implementing explicit `for` loops (over m examples or n features) is computationally inefficient in deep learning.

- We will use **Vectorization** to process the entire training set without explicit loops to handle large datasets efficiently.
