## **Neural Networks Intuition**

### **The Biological Motivation 🧠**

- **Initial Goal:** Neural networks were originally invented in the 1950s with the goal of mimicking the human brain.

- **The Biological Neuron:** A biological neuron receives input signals via **dendrites**, processes them in its cell body (nucleus), and sends an output signal (an electrical impulse) down an **axon**.1

- **Modern Context:** This analogy is now considered very loose. Our understanding of the brain is still limited, and modern neural networks are built using engineering principles, not by directly copying biology.

### **The Artificial Neuron**

- An **artificial neuron** is a highly simplified mathematical model of its biological counterpart.2

- **Function:**
  1. It receives one or more numerical **inputs**.
  2. It performs a simple **computation**.
  3. It produces a single numerical **output**.

- A **neural network** is created by connecting many of these simple neurons, where the outputs from one layer of neurons become the inputs for the next layer.3

---

### **A Brief History of Neural Networks**

Neural networks have gone through several cycles of popularity and decline (sometimes called "AI winters").4

1. **1950s:** Initial invention.

2. **1980s-1990s:** A resurgence in popularity, showing success in applications like handwritten digit recognition (e.g., reading postal codes).5

3. **Late 1990s:** Fell out of favor again.

4. **~2005-Present:** A massive resurgence, rebranded as **Deep Learning**.
   - This new wave first revolutionized **speech recognition**.6

   - It then had a major impact on **computer vision** (e.g., the 2012 ImageNet competition).7

   - It subsequently transformed **Natural Language Processing (NLP)** and many other fields.8

---

### **Why Are Neural Networks Successful Now? 🚀**

Two primary factors are responsible for the recent "deep learning revolution":

1. **Massive Data Availability ("Big Data")**
   - Our digital society (internet, mobile phones) generates enormous amounts of data.9

   - **Traditional algorithms** (like linear or logistic regression) **plateau** in performance. Their accuracy stops improving even when fed more data.

   - **Neural networks scale with data.** Larger neural networks can effectively learn from and leverage massive datasets to achieve much higher performance.10

2. **Faster Computation (Hardware)**
   - Training large neural networks is computationally very intensive.

   - The rise of powerful **GPUs (Graphics Processing Units)**, originally built for video games, provided the perfect hardware for the parallel computations required by neural networks.11

