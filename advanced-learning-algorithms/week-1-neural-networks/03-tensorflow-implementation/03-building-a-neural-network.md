## **Building a Neural Network in TensorFlow**

While you can compute the output of a network one layer at a time, TensorFlow provides a much simpler and more powerful way to build, train, and run models using the **`Sequential` API**.

---

### **The `Sequential` Model: A Simpler Approach**

Instead of manually passing the output of one layer to the input of the next, the `Sequential` model allows you to define a "stack" or sequence of layers. TensorFlow then handles the work of stringing them together.

- **Old Method (Manual Forward Prop):**

  ```python
  a1 = Layer1(x)
  a2 = Layer2(a1)
  a3 = Layer3(a2)
  ```

- **New Method (Using `Sequential`):**
  You define all the layers at once, and TensorFlow connects them for you.

---

### **A Common Coding Convention 🧑‍💻**

The most common way to build a model is to define the layers directly _inside_ the `Sequential` function.

- **Example (Digit Classification):**

  ```python
  import tensorflow as tf
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.models import Sequential

  model = Sequential([
      Dense(units=25, activation='sigmoid'),  # Layer 1
      Dense(units=15, activation='sigmoid'),  # Layer 2
      Dense(units=1, activation='sigmoid')   # Layer 3 (Output)
  ])
  ```

- This single `model` object now contains the entire three-layer network architecture.

---

### **How to Use the `Sequential` Model**

Once you have a `Sequential` model, training it and making predictions becomes simple.

- **1. Training the Model (Preview):**
  - **`model.compile(...)`**: A required setup step to configure the model for training (details will be covered next week).
  - **`model.fit(X, Y)`**: A single command to **train** the network. You provide the training data (`X`) and the target labels (`Y`), and TensorFlow handles the entire gradient descent process.

- **2. Making Predictions (Inference):**
  - **`model.predict(X_new)`**: This is the new way to perform forward propagation. You pass in a new input (`X_new`), and the model automatically runs it through all the layers and returns the final prediction.

---

### **Understanding "Under the Hood"**

- While frameworks like TensorFlow allow you to build complex neural networks in just a few lines of code, it is critical to understand _what_ those lines are actually doing.
- Most engineers use libraries, but the best engineers understand the underlying mechanics. To build this understanding, the next step is to learn how to implement forward propagation from scratch.
