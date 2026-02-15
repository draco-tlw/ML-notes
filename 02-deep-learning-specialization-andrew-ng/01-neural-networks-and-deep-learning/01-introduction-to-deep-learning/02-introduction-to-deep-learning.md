# Note: Foundations and Drivers of Deep Learning

**Source:** Coursera - Neural Networks and Deep Learning (Weeks 1)

## 1. What is a Neural Network?

**Intuition: The Housing Price Prediction Example**

- **Goal:** Predict the price of a house (y) based on its size (x).
- **Linear Regression approach:** A straight line fit.
- **Constraint:** Prices cannot be negative.
- **The ReLU Function:**
- The function creates a "bend" at zero. It is zero for negative values and linear for positive values.
- **Definition:** **ReLU** stands for **Rectified Linear Unit**. "Rectify" implies taking the max of 0.
- Mathematically: f(x) = max(0, x).

- **The Neuron Analogy:**
- A single ReLU unit can be thought of as a single neuron.
- Input (Size) \rightarrow Neuron (computes linear function, applies max(0)) \rightarrow Output (Price).

**Building a Network**

- **Stacking:** A larger neural network is formed by stacking many single neurons together (like Lego bricks).
- **Example with Multiple Features:**
- **Input Layer (x):** Size, # Bedrooms, Zip Code, Wealth.
- **Hidden Units:** The network calculates intermediate features automatically (e.g., Family Size, Walkability, School Quality).
- **Output Layer (y):** Price.

- **Dense Connectivity:** In a standard network, every input feature is connected to every hidden unit. The network autonomously decides which features matter for each node; we do not manually define "Family Size."

## 2. Supervised Learning with Neural Networks

Currently, the vast majority of economic value created by Neural Networks comes from **Supervised Learning** (learning a mapping from Input x to Output y).

### Common Applications

| Application             | Input (x)        | Output (y)           | Application Type |
| ----------------------- | ---------------- | -------------------- | ---------------- |
| **Real Estate**         | Home Features    | Price                | Standard NN      |
| **Online Advertising**  | Ad & User Info   | Click (0/1)          | Standard NN      |
| **Photo Tagging**       | Image            | Object ID (1...1000) | CNN              |
| **Speech Recognition**  | Audio Clip       | Text Transcript      | RNN              |
| **Machine Translation** | English Sentence | Chinese Sentence     | RNN              |
| **Autonomous Driving**  | Image, Radar     | Car Positions        | Hybrid / Custom  |

### Neural Network Architectures

Different data types require different standard architectures:

- **Standard NN:** Used for general datasets (Real Estate, Ads).
- **CNN (Convolutional Neural Networks):** Used primarily for **Image** data.
- **RNN (Recurrent Neural Networks):** Used for **Sequence** data (Audio, Language/NLP) because these possess a temporal component.

### Data Types

- **Structured Data:** Databases where features have well-defined meanings (e.g., columns for Age, Price, ID). Historically, companies have created significant value here (e.g., ad systems).
- **Unstructured Data:** Raw data types like Audio, Images, and Text.
- _Note:_ Humans are naturally good at processing unstructured data. A major recent breakthrough is that Deep Learning now allows computers to process this data effectively as well.

## 3. Why is Deep Learning Taking Off Now?

The basic technical ideas have existed for decades, but three specific drivers have fueled recent growth:

### A. Data (Scale)

- **Digitization:** The shift to digital devices (mobile, web, IoT) has generated massive amounts of data.
- **Notation:** We use lowercase **m** to denote the size of the training set (number of examples).
- **Performance vs. Data Scale:**
- **Traditional Algorithms (SVM, Logistic Regression):** Performance plateaus even as data increases; they cannot utilize massive datasets effectively.
- **Neural Networks:** Performance scales with data.
- Small NN \rightarrow Better performance.
- Medium NN \rightarrow Even better.
- Large NN \rightarrow Best performance (keeps rising).

- **Conclusion:** To achieve high performance, you generally need **Large Neural Networks** + **Huge Amounts of Data**.

### B. Computation

- The rise of **GPUs** and faster hardware has enabled the training of massive networks that were previously impossible to calculate.

### C. Algorithmic Innovation

- Innovations are often focused on making code run **faster** to improve iteration speed.
- **Key Example: Sigmoid vs. ReLU**
- **Sigmoid Function:** Has regions where the slope (gradient) is nearly zero. This causes learning to become very slow (vanishing gradient problem).
- **ReLU Function:** The gradient is equal to 1 for all positive values. This prevents the gradient from shrinking to zero, allowing **Gradient Descent** to work much faster.

- **The Iterative Cycle:** Idea \rightarrow Code \rightarrow Experiment.
- Faster computation/algorithms reduce the time for one loop of this cycle.
- Reduced turnaround time (from months to minutes) allows researchers to test more ideas, leading to faster progress.
