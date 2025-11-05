## **Multiclass Classification**

**Multiclass classification** is a type of classification problem where the output variable, `y`, can be one of **three or more** possible discrete categories (or classes).

This is an extension of binary classification, where `y` could only be 0 or 1.

---

### **Examples of Multiclass Classification 🗂️**

- **Handwritten Digit Recognition:** Classifying an image as one of the 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9).
- **Medical Diagnosis:** Identifying which of several possible diseases a patient has (e.g., "Disease A," "Disease B," "Disease C," or "Healthy").
- **Manufacturing Defects:** Inspecting a product and classifying its defect type (e.g., "Scratch," "Chip," "Discoloration," or "No Defect").

---

### **Contrast with Binary Classification**

- **Binary Classification:**
  - Has two output classes (e.g., 0 or 1).
  - The model (logistic regression) estimates a single probability: $P(y=1)$.
- **Multiclass Classification:**
  - Has `N` (where $N > 2$) output classes (e.g., 1, 2, 3, 4).
  - The model must estimate the probability for **each class**: $P(y=1)$, $P(y=2)$, $P(y=3)$, and $P(y=4)$.
  - The model learns decision boundaries that divide the input space into multiple regions, one for each class.

---

### **Softmax Regression: The Algorithm for Multiclass**

To handle multiclass problems, we need a new algorithm. **Softmax regression** is a generalization of logistic regression that can handle more than two classes. This will be the key component used to build neural networks for multiclass classification.
