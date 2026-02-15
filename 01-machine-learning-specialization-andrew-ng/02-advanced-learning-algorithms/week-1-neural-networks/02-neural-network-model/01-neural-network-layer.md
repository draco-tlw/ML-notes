## **The Building Block: A Layer of Neurons**

A neural network is constructed by stacking layers of neurons. Each layer takes a vector of numbers as input and produces a vector of numbers as output, which is then passed to the next layer.

---

### **Inside a Hidden Layer (e.g., Layer 1)**

Let's look at a single hidden layer that takes 4 input features (`x`) and has 3 neurons.

- **Neuron Computation:** Each neuron in this layer functions as an independent **logistic regression unit**. It has its own set of parameters (`w` and `b`) and computes its own output value.
- **Neuron 1:**
  - Computes its activation $a_1 = g(\vec{w}_1 \cdot \vec{x} + b_1)$.
  - $g(z)$ is the sigmoid (logistic) function: $g(z) = \frac{1}{1 + e^{-z}}$.
- **Neuron 2:**
  - Computes its activation $a_2 = g(\vec{w}_2 \cdot \vec{x} + b_2)$.
- **Neuron 3:**
  - Computes its activation $a_3 = g(\vec{w}_3 \cdot \vec{x} + b_3)$.
- **Layer Output:** The outputs of these three neurons (e.g., [0.3, 0.7, 0.2]) are collected into a vector. This is the **activation vector** of the layer.

---

### **New Notation: Layers ✍️**

To keep track of parameters and activations in different layers, we use a **superscript square bracket `[L]`** to denote the layer number.

- **Input Layer:** This is considered **Layer 0**. Its output is just the input feature vector $\vec{x}$.
- **First Hidden Layer (Layer 1):**
  - The parameters for the neurons in this layer are $\vec{w}^{[1]}$ and $b^{[1]}$.
  - The output activation vector of this layer is $\vec{a}^{[1]}$.
- **Second Layer (e.g., Output Layer, Layer 2):**
  - The parameters for this layer are $\vec{w}^{[2]}$ and $b^{[2]}$.
  - The output activation of this layer is $\vec{a}^{[2]}$.

---

### **The Output Layer (e.g., Layer 2)**

The output layer is just another layer that follows the same rules.

- **Input:** It takes the activation vector from the previous layer ($\vec{a}^{[1]}$) as its input.
- **Computation:** It performs its own logistic regression calculation.
  - $z^{[2]} = \vec{w}^{[2]} \cdot \vec{a}^{[1]} + b^{[2]}$
  - $\vec{a}^{[2]} = g(z^{[2]})$
- **Final Output:** The activation of the output layer, $\vec{a}^{[2]}$, is the neural network's final prediction (e.g., 0.84). This represents the probability that $y=1$.

### **Final Prediction (Optional Thresholding)**

If you need a binary (0 or 1) prediction, you can apply a threshold to the final probability, just like in logistic regression.

- **Rule:**
  - If $\vec{a}^{[2]} \ge 0.5$, predict $\hat{y} = 1$.
  - If $\vec{a}^{[2]} < 0.5$, predict $\hat{y} = 0$.
