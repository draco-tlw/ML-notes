## **Anomaly Detection 🕵️**

This is an unsupervised learning algorithm that examines an unlabeled dataset of "normal" events and learns to detect or "flag" unusual or anomalous events.

### **The Problem**

Imagine you are a manufacturer of aircraft engines.

- You produce many engines, and you want to ensure they are reliable.
- You can measure features for each engine, such as:
  - $x_1$: Heat generated
  - $x_2$: Vibration intensity
- **Dataset:** You have a dataset of $m$ engines that you have manufactured. Most of these are normal, good engines.
- **Goal:** When a new engine ($x_{test}$) is made, you want to know: "Is this engine similar to the ones I've seen before (normal), or is it strange (anomalous)?"

### **How it Works: Density Estimation**

The core technique used for anomaly detection is called **Density Estimation**.

1.  **Build a Model ($p(x)$):**
    - The algorithm uses the training data to build a probability model $p(x)$.
    - This model tells you the probability of seeing a specific set of features.
    - Values of $x$ that are "normal" (seen often) will have a **high probability**.
    - Values of $x$ that are "unusual" (rare) will have a **low probability**.

2.  **Set a Threshold ($\epsilon$):**
    - You choose a small number $\epsilon$ (epsilon) as your cutoff threshold.

3.  **Detect Anomalies:**
    - For a new example $x_{test}$, calculate $p(x_{test})$.
    - **If $p(x_{test}) < \epsilon$:** The probability is very low. Flag it as an **anomaly**.
    - **If $p(x_{test}) \ge \epsilon$:** The probability is high enough. It is **normal**.

---

### **Real-World Applications**

Anomaly detection is extremely common in industry, used for ensuring quality and security.

- **Fraud Detection:**
  - **Context:** Websites tracking user behavior (login frequency, typing speed, transaction volume).
  - **Usage:** Model $p(x)$ for a user's typical behavior. If a user's activity suddenly has a very low probability (e.g., logging in from a new country and typing much faster), flag it for security review (e.g., ask for 2-factor authentication).
  - **Financial Fraud:** Identifying unusual credit card transactions.

- **Manufacturing (Quality Control):**
  - **Context:** Producing items like aircraft engines, smartphones, or circuit boards.
  - **Usage:** Measure features of every unit. If a unit has anomalous measurements (e.g., high vibration), inspect it for defects before shipping.

- **Monitoring Data Centers:**
  - **Context:** Managing thousands of computer servers.
  - **Usage:** Monitor features like memory usage, CPU load, and network traffic. If a machine behaves anomalously (e.g., CPU load spikes while network traffic is zero), it might be broken, overheating, or hacked.
