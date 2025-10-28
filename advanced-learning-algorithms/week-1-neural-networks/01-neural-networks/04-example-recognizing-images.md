## **Neural Networks for Computer Vision**

Neural networks are particularly powerful for computer vision tasks, such as face recognition, because of their ability to learn a hierarchy of features directly from pixel data.

### **Problem Setup: Face Recognition 📸**

- **Input ($\vec{x}$):** An image. A computer sees an image as a grid of pixel brightness values. For example, a 1000x1000 pixel image becomes a 1,000,000-element feature vector (1000 \* 1000 = 1,000,000).
- **Output ($\hat{y}$):** The identity of the person in the picture.

### **Hierarchical Feature Learning**

A deep neural network (one with multiple hidden layers) learns to detect features in a hierarchical way. Earlier layers learn simple features, and later layers combine them into progressively more complex ones.

- **First Hidden Layer:** Learns to detect simple, small-scale features. It's like giving each neuron a small window of the image.
  - _Example:_ One neuron might learn to find small vertical edges, another might find horizontal edges, and another might find edges at a 45-degree angle.

- **Second Hidden Layer:** It takes the simple edge features from the first layer as its input and learns to combine them into more complex _parts_.
  - _Example:_ By combining various lines and edges, neurons in this layer might learn to detect an **eye**, a **nose**, or the curve of an **ear**.

- **Third Hidden Layer:** It takes the parts from the second layer as input and learns to combine them into even larger shapes.
  - _Example:_ Neurons here might learn to detect different **face shapes** (e.g., a round face, an oval face).

- **Output Layer:** It takes the high-level features from the final hidden layer (like "oval face," "has a nose," "has two eyes") and uses them to make the final prediction of the person's identity.

### **Automatic Feature Learning ✅**

The most remarkable property of a neural network is that **it learns this entire feature hierarchy automatically**.

- We do _not_ need to tell the network to look for edges, eyes, or noses. We only provide it with the raw pixel data (input) and the correct labels (output).
- During training, the network figures out on its own that detecting edges is useful for detecting eyes, and detecting eyes is useful for identifying a face.
- This same algorithm, if trained on a different dataset (like pictures of cars), would automatically learn a different hierarchy (e.g., edges $\rightarrow$ wheels, windows $\rightarrow$ full car shapes).
