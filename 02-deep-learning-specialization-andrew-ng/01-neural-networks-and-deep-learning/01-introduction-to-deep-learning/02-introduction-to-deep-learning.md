# Introduction to Neural Networks

### 1. Definition and Basic Intuition

- **Deep Learning:** Refers to the process of training Neural Networks (often very large ones).
- **The Single Neuron (Simplest NN):**
  - Can be visualized using a **Linear Regression** model.
  - **Example:** Predicting housing prices ($y$) based on size ($x$).
  - Instead of a standard straight line (which might predict negative prices), the function is "bent" to ensure the output is never less than zero.
  - **Mechanism:** Input ($x$) $\rightarrow$ Neuron (Computation) $\rightarrow$ Output ($y$).

### 2. The ReLU Function

- **Name:** Rectified Linear Unit (**ReLU**).
- **Function:** $f(x) = max(0, x)$.
- **Characteristics:**
  - Output is 0 when the input is negative.
  - Output increases linearly when the input is positive.
  - This is a fundamental building block (activation function) in neural networks.

### 3. Building Larger Networks

- **Stacking Neurons:** A larger neural network is formed by stacking multiple single neurons together (analogous to stacking Lego bricks).
- **Complex Inputs:** Networks can handle multiple input features simultaneously.
  - _Example Inputs:_ Size, Number of Bedrooms, Zip Code, Wealth.
- **Hidden Layers & Feature Extraction:**
  - Intermediate nodes (circles between input and output) are called **Hidden Units**.
  - While humans might conceptually map these units to features like "Family Size" or "Walkability," the network self-determines what each node represents.

### 4. Architecture and Connectivity

- **Dense Connections:**
  - In a standard neural network layer, the input layer is **densely connected** to the hidden layer.
  - **Rule:** _Every_ input feature connects to _every_ hidden unit.
  - The network is given the input ($x$) and the target output ($y$) and learns the intermediate mappings/features automatically.

### 5. Application Context

- **Supervised Learning:**
  - Neural networks are currently most powerful in **Supervised Learning** contexts.
  - **Goal:** Accurately map Input ($x$) to Output ($y$) using labeled training data.

---

# Supervised Learning with Neural Networks

### 1\. Overview

- **Economic Impact:** The vast majority of economic value currently created by Neural Networks comes from **Supervised Learning**.
- **Core Function:** The goal is to learn a function that maps an Input ($x$) to an Output ($y$).

### 2\. Applications of Supervised Learning

Neural networks are effectively applied across various industries by cleverly selecting $x$ and $y$.

| Application             | Input ($x$)        | Output ($y$)           | Note                |
| :---------------------- | :----------------- | :--------------------- | :------------------ |
| **Real Estate**         | Home features      | Price                  | Standard prediction |
| **Online Advertising**  | Ad info, User info | Click (0/1)            | Highly lucrative    |
| **Photo Tagging**       | Image              | Index (1-1000)         | Computer Vision     |
| **Speech Recognition**  | Audio clip         | Text transcript        | Audio processing    |
| **Machine Translation** | English sentence   | Chinese sentence       | NLP                 |
| **Autonomous Driving**  | Image, Radar info  | Position of other cars | Hybrid system       |

### 3\. Neural Network Architectures

Different data types require specific Neural Network (NN) architectures for optimal performance.

- **Standard Neural Networks:**
  - Used for general tabular data.
  - _Examples:_ Real Estate, Online Advertising.
- **Convolutional Neural Networks (CNN):**
  - Specialized for **Image Data**.
  -

```
* *Examples:* Photo tagging, Object detection.
```

- **Recurrent Neural Networks (RNN):**
  - Specialized for **Sequence Data** (data with a temporal component).
  -

```
* *Examples:* Audio (1D time series), Language/Text (sequence of words).
```

- **Hybrid Architectures:**
  - Used for complex tasks combining different data types.
  - _Example:_ Autonomous driving (Images + Radar).

### 4\. Structured vs. Unstructured Data

Machine Learning is applied to two distinct categories of data.

- **Structured Data:**
  - **Definition:** Data found in databases with well-defined meanings for each feature (column).
  - _Examples:_ Housing prices database (Size, \#Bedrooms), User data (Age, Ad ID).
  - _Economic Value:_ Creates significant short-term economic value (e.g., better ad systems, recommendations).
- **Unstructured Data:**
  - **Definition:** Raw data types where features are not explicitly defined as columns.
  - _Examples:_ Audio (raw wave), Images (pixel values), Text (individual words).
  - _Human vs. Machine:_ Humans are naturally empathetic and good at understanding unstructured data. Historically, computers struggled with this.
  - _Impact of Deep Learning:_ Neural Networks have drastically improved computer performance on unstructured data (Speech recognition, Image recognition), enabling new applications.

---

# Why is Deep Learning Taking Off?

### 1. The Scale Driver: Data

The primary driver for the recent surge in Deep Learning performance is the massive increase in available data.

- **Performance vs. Data Graph:**
  - **Traditional Algorithms (e.g., SVM, Logistic Regression):** Performance improves with data up to a point but eventually **plateaus**. They cannot effectively utilize huge amounts of data.
  - **Neural Networks:** Performance continues to increase as the amount of data increases.
  - **Network Size:**
    - **Small NN:** Performance is better than traditional algorithms but may still plateau early.
    - **Medium NN:** Better performance.
    - **Large NN:** Dominates performance on very large datasets; rarely plateaus.

- **Sources of Data:**
  - Digitization of society (web activity, mobile apps).
  - Inexpensive sensors (cameras, accelerometers, IoT).
- **Notation:**
  - $m$: The size of the training set (number of labeled training examples).
- **Data Regimes:**
  - **Small Data Regime (Low $m$):** The ordering of algorithms is undefined. Performance depends heavily on **feature engineering** skill rather than the algorithm itself (an SVM might beat a Neural Network here).
  - **Big Data Regime (High $m$):** Large Neural Networks consistently dominate other approaches.

### 2. The Computation Driver

- **Hardware:** The ability to train very large neural networks has been enabled by faster **CPUs** and specialized hardware like **GPUs**.
- **Impact:** Computation power allows us to scale up the size of the neural networks (number of hidden units, parameters, and connections).

### 3. The Algorithmic Driver

Algorithmic innovation has largely focused on making neural networks run **faster**, which indirectly enables the use of more data and larger networks.

- **Key Example: Activation Functions**
  - **Sigmoid Function:** Historically used but problematic. It has regions where the slope (gradient) is nearly zero, causing learning to be very slow (the Vanishing Gradient problem).
  - **ReLU (Rectified Linear Unit):**
    - Gradient is 1 for all positive input values.
    - Gradient is much less likely to shrink to zero.
    - **Result:** Switching from Sigmoid to ReLU makes **Gradient Descent** work much faster.

### 4. The Iterative Cycle of Research

Fast computation is crucial not just for the final result, but for the **research workflow**.

- **The Cycle:** Idea $\rightarrow$ Code $\rightarrow$ Experiment $\rightarrow$ Result.
- **Productivity:**
  - Slow computation (e.g., waiting a month for a result) hinders progress.
  - Fast computation (e.g., results in 10 minutes or a day) allows practitioners to iterate quickly, test more ideas, and discover effective architectures faster.

### Summary of Drivers

The rise of Deep Learning is fueled by a positive feedback loop of:

1.  **Data:** Increasing digitization and sensor availability.
2.  **Computation:** Specialized hardware (GPUs).
3.  **Algorithms:** Innovations that optimize speed (e.g., ReLU).

---
