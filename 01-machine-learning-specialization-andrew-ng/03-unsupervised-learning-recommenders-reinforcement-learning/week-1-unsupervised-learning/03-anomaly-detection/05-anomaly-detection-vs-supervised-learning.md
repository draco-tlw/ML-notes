## **Anomaly Detection vs. Supervised Learning ⚖️**

When you have a dataset with very few positive examples ($y=1$) and a large number of negative examples ($y=0$), the choice between using Anomaly Detection or Supervised Learning can be subtle.

Here is the framework for deciding which to use.

### **1. Based on Number of Examples**

The most immediate indicator is the volume of positive data available.

| **Anomaly Detection**                                                                                                                            | **Supervised Learning**                                                                                           |
| :----------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| **Very small** number of positive examples ($y=1$). (e.g., 0 to 20 examples).                                                                    | **Large** number of positive examples ($y=1$).                                                                    |
| **Large** number of negative examples ($y=0$).                                                                                                   | **Large** number of negative examples ($y=0$).                                                                    |
| The algorithm learns $p(x)$ using **only** the negative examples. The few positive examples are reserved for the Cross-Validation and Test sets. | The algorithm uses **both** positive and negative examples during training to learn the boundary between classes. |

### **2. Based on the Nature of the "Anomaly"**

The more important distinction lies in whether future anomalies will look like past anomalies.

#### **Use Anomaly Detection When:**

- There are many **different types** of anomalies.
- It is hard for an algorithm to learn what an anomaly "looks like" from a small set of positive examples.
- **Crucial:** Future anomalies may look **nothing like** the anomalies you have seen so far.
- **Method:** The algorithm builds a model of "Normal." Anything that deviates from "Normal" is flagged.

#### **Use Supervised Learning When:**

- There are enough positive examples for the algorithm to get a sense of what they look like.
- **Crucial:** Future positive examples are likely to be **similar** to the positive examples in the training set.
- **Method:** The algorithm learns a boundary that specifically separates positive examples from negative examples.

---

### **Real-World Examples**

| Application                         | Recommended Algorithm   | Reason                                                                                                                                    |
| :---------------------------------- | :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **Financial Fraud**                 | **Anomaly Detection**   | Hackers constantly invent **new** methods of fraud. A new attack might look completely different from a previous one.                     |
| **Email Spam**                      | **Supervised Learning** | Spam often follows predictable patterns (selling drugs, phishing links). Future spam usually resembles past spam.                         |
| **Manufacturing (Unknown Defects)** | **Anomaly Detection**   | Example: Aircraft engines. A failure might happen for a brand new reason (a new type of crack or material failure) never seen before.     |
| **Manufacturing (Known Defects)**   | **Supervised Learning** | Example: Scratched smartphone screens. You have seen thousands of scratches; you just want to find _more of the same_ defect.             |
| **Data Center Security**            | **Anomaly Detection**   | Hackers find new exploits (Zero-day attacks). The machine behaves in a "weird" way that hasn't been defined yet.                          |
| **Weather / Medical**               | **Supervised Learning** | Diagnosing a specific disease or predicting rain vs. sun. The "positive" cases (having the disease/rain) are well-defined and repetitive. |

### **Summary**

- **Anomaly Detection:** "I don't know what might go wrong, but I know what 'normal' looks like. Alert me if _anything_ weird happens."
- **Supervised Learning:** "I know exactly what goes wrong (or what I'm looking for). Here are 1,000 examples of it. Find me more of _this specific thing_."
