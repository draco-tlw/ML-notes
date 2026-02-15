## **Supervised Learning**

**Supervised learning** is the most common type of machine learning and is responsible for most of its economic value today. It's all about learning a mapping from an **input (x)** to an **output (y)** based on labeled examples.

### **How it Works 🧠**

The core idea is to "supervise" the algorithm by giving it a dataset that includes the "right answers."

1. **Training:** The algorithm is fed a large amount of training data, where each data point consists of an input `x` and its corresponding correct output label `y`.
2. **Learning:** By seeing many `(x, y)` pairs, the algorithm learns the underlying relationship between the inputs and outputs.
3. **Prediction:** Once trained, the model can be given a **new input `x`** that it has never seen before and produce a prediction, or a good guess, for the output `y`.

---

### **Examples of Supervised Learning**

| Application             | Input (x)             | Output (y)                    |
| :---------------------- | :-------------------- | :---------------------------- |
| **Spam filtering**      | An email              | Spam (1) or Not Spam (0)      |
| **Speech recognition**  | An audio clip         | Text transcript               |
| **Machine translation** | A sentence in English | The sentence in Spanish       |
| **Online advertising**  | Ad info + user info   | Will the user click? (Yes/No) |
| **Self-driving cars**   | Image from a camera   | Position of other cars        |
| **Visual inspection**   | Picture of a phone    | Defect (1) or No Defect (0)   |

---

### **Types of Supervised Learning: Regression**

One of the two major types of supervised learning is called **regression**.

- **Goal:** To predict a **continuous numerical value**. This means the output can be any number within a range.
- **Example: Housing Price Prediction**
  - You provide the model with a dataset of houses, where the input `x` is the **size of the house** (e.g., in square feet) and the output `y` is its **selling price**.
  - The algorithm learns the relationship by fitting a line or a curve to this data.
  - It can then predict a price for a house size it has never seen before.
