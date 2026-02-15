## **Implementing Content-Based Filtering in TensorFlow 💻**

This video walks through the specific code structure for building the "Two-Tower" neural network architecture using the TensorFlow Keras API.

### **1. Defining the Networks**

We create two separate Sequential models: one for the user and one for the item (movie). Both must end with an output layer of the same size (e.g., 32 units).

**The User Network:**

```python
user_NN = tf.keras.models.Sequential([
    # Hidden layers with Relu activation
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer (the "User Vector")
    tf.keras.layers.Dense(32)
])
```

**The Item (Movie) Network:**

```python
item_NN = tf.keras.models.Sequential([
    # Hidden layers with Relu activation
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer (the "Item Vector")
    tf.keras.layers.Dense(32)
])
```

---

### **2. The Forward Pass and Normalization**

We define the input layers, pass them through the networks to get the vectors ($v_u$ and $v_m$), and then apply a crucial normalization step.

**Why Normalize?**
Normalizing the vectors to have a length of 1 (L2 norm) significantly improves the algorithm's performance.

```python
# 1. Create Inputs
input_user = tf.keras.layers.Input(shape=(num_user_features))
input_item = tf.keras.layers.Input(shape=(num_item_features))

# 2. Compute Vectors (Pass inputs through the networks)
vu = user_NN(input_user)
vm = item_NN(input_item)

# 3. L2 Normalization (Scale vectors to length 1)
vu = tf.linalg.l2_normalize(vu, axis=1)
vm = tf.linalg.l2_normalize(vm, axis=1)
```

---

### **3. The Dot Product and Model Definition**

Finally, we calculate the predicted rating by taking the dot product of the two normalized vectors.

```python
# 4. Compute Dot Product (The Prediction)
# Keras has a specific layer for this operation
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# 5. Define the Full Model
# Inputs: User features AND Item features
# Output: The dot product prediction
model = tf.keras.Model([input_user, input_item], output)

# 6. Compile the Model
# Use Mean Squared Error cost function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_squared_error')
```

---

### **Summary of Course 3, Week 2**

You have now completed the deep dive into Recommender Systems\!

- **Collaborative Filtering:** Recommendations based on user/item interactions (ratings).
- **Content-Based Filtering:** Recommendations based on user/item features using Neural Networks.
- **Architecture:** Retrieval (fast candidate generation) $\rightarrow$ Ranking (precise scoring).
- **Ethics:** The importance of optimizing for user well-being, not just profit or engagement.
