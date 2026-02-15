# Note: Shallow Neural Networks

**Source:** Coursera - Neural Networks and Deep Learning (Week 3)

## 1. Neural Network Representation

A "Shallow" Neural Network typically refers to a network with one or a few hidden layers.

- **Architecture Terminology:**
- **Input Layer (Layer 0):** Contains the input features x.
- Notation: a^{[0]} = x.

- **Hidden Layer (Layer 1):** The nodes in the middle that are not directly observed in the training set.
- Notation: a^{[1]} (activations of layer 1).

- **Output Layer (Layer 2):** The final node(s) that generate the prediction \hat{y}.
- Notation: a^{[2]} = \hat{y}.

- **Counting Layers:** By convention, we do not count the input layer. A network with one hidden layer and one output layer is called a **2-Layer Neural Network**.
- **Notation Standard:**
- Superscript [l] denotes quantities associated with the l^{th} layer (e.g., W^{[1]}, b^{[1]}).
- Superscript (i) denotes the i^{th} training example (e.g., x^{(i)}).
- Subscript n denotes the index of a unit in a layer.

## 2. Computing the Output (Forward Propagation)

A neural network basically repeats the Logistic Regression computation multiple times.

### Step-by-Step for a Single Example x^{(i)}

For a network with one hidden layer:

1. **Layer 1 (Hidden):**

- z^{[1]} = W^{[1]}x + b^{[1]}
- a^{[1]} = \sigma(z^{[1]})

2. **Layer 2 (Output):**

- z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}
- a^{[2]} = \sigma(z^{[2]}) (assuming binary classification)

### Vectorizing Across m Examples

We stack examples in columns to form matrices X, Z, and A.

- **Matrix Dimensions:**
- X: (n_x, m)
- Z^{[l]}, A^{[l]}: (n^{[l]}, m) where n^{[l]} is the number of units in layer l.

- **Vectorized Equations:**

1. Z^{[1]} = W^{[1]}X + b^{[1]} (Python broadcasts b^{[1]})
2. A^{[1]} = g^{[1]}(Z^{[1]})
3. Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}
4. A^{[2]} = g^{[2]}(Z^{[2]})

## 3. Activation Functions

The choice of function g(z) significantly affects performance.

| Function       | Formula            | Range             | Pros/Cons                                                                                                                                      | Use Case                                                               |
| -------------- | ------------------ | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Sigmoid**    | \frac{1}{1+e^{-z}} | (0, 1)            | **Con:** Slope is near 0 for large/small z (slow learning). Mean is 0.5 (not centered).                                                        | **Output Layer** (Binary Classification only). Avoid in hidden layers. |
| **Tanh**       | \tanh(z)           | (-1, 1)           | **Pro:** Data is centered (mean \approx 0), making learning easier for next layer. **Con:** Still suffers from vanishing gradient at extremes. | **Hidden Layers** (Superior to Sigmoid).                               |
| **ReLU**       | \max(0, z)         | [0, \infty)       | **Pro:** Fast learning; derivative is 1 for z>0 (no vanishing gradient). **Con:** Derivative is 0 for z<0 ("Dead ReLU").                       | **Default choice** for Hidden Layers.                                  |
| **Leaky ReLU** | \max(0.01z, z)     | (-\infty, \infty) | **Pro:** Fixes "Dead ReLU" problem by having a slight slope for negative values.                                                               | Hidden Layers (Alternative to ReLU).                                   |

### Why Non-Linear Activation Functions?

- **Identity Activation:** If you use a linear activation function (i.e., g(z) = z), the neural network just outputs a linear function of the input.
- **Collapse:** A deep network with only linear activations collapses into a single linear regression model. The composition of linear functions is still linear.
- **Exception:** You might use a linear activation in the **Output Layer** if you are doing regression (predicting real numbers like house prices), but hidden layers must still be non-linear.

## 4. Gradient Descent (Backpropagation)

To train the network, we compute derivatives of the Cost Function J with respect to parameters W and b.

### Forward Propagation (Recap)

- Z^{[1]} = W^{[1]}X + b^{[1]}
- A^{[1]} = g^{[1]}(Z^{[1]})
- Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}
- A^{[2]} = g^{[2]}(Z^{[2]})

### Backward Propagation Formulas

Derivatives are computed right-to-left (Layer 2 \rightarrow Layer 1).

- **Layer 2 (Output):**
- dZ^{[2]} = A^{[2]} - Y
- dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}
- db^{[2]} = \frac{1}{m} \text{np.sum}(dZ^{[2]}, \text{axis}=1, \text{keepdims=True})

- **Layer 1 (Hidden):**
- dZ^{[1]} = W^{[2]T} dZ^{[2]} \* g^{[1]'}(Z^{[1]})
- _Note:_ `*` denotes element-wise multiplication.
- _Note:_ g^{[1]'} is the derivative of the hidden activation function.

- dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T
- db^{[1]} = \frac{1}{m} \text{np.sum}(dZ^{[1]}, \text{axis}=1, \text{keepdims=True})

### Derivatives of Activation Functions (g'(z))

- **Sigmoid:** g'(z) = a(1-a)
- **Tanh:** g'(z) = 1 - a^2
- **ReLU:** 0 if z<0, 1 if z>0.

## 5. Random Initialization

**The Symmetry Problem:**

- If you initialize all weights W to **zero**, all hidden units will calculate the exact same function.
- a^{[1]}\_1 = a^{[1]}\_2 = \dots
- Backpropagation will compute identical derivatives for all hidden units (dW rows will be identical).
- The network will fail to learn distinct features.

**The Solution:**

- Initialize W to **small random values**.
- `W = np.random.randn((n1, n0)) * 0.01`

- Initialize b to **zero** (this is okay because W breaks the symmetry).

**Why Small Values?**

- If weights are too large, z will be very large or very small.
- This pushes activations into the "flat" regions of Tanh or Sigmoid functions where the slope (gradient) is nearly zero.
- This leads to slow learning (Vanishing Gradient problem).

---

**Next Step:**
I have summarized the mechanics of Shallow Neural Networks. This concludes the material for Week 3.

Would you like clarification on the **Backpropagation formulas** (specifically the dimensions or the element-wise multiplication), or are you ready to move on to **Week 4: Deep Neural Networks**?
