## **The Precision-Recall Trade-off (Optional) ⚖️**

Ideally, we want a model to have both high precision and high recall. However, in practice, these two metrics are often in conflict: improving one can hurt the other. This relationship is managed by adjusting the model's **classification threshold**.

---

### **The Role of the Threshold**

Recall that logistic regression outputs a probability $f(x)$ (a number between 0 and 1). We typically use a **threshold of 0.5** to make a prediction:

- If $f(x) \ge 0.5$, predict $\hat{y} = 1$
- If $f(x) < 0.5$, predict $\hat{y} = 0$

We can _change_ this threshold to prioritize either precision or recall, depending on our goal.

---

### **Two Scenarios for Adjusting the Threshold**

#### **Scenario 1: Prioritizing High Precision (Being "Confident")**

- **Goal:** We want to be very sure that when the model predicts $\hat{y}=1$, it is correct. We want to avoid **False Positives (FP)**.
- **Example:** Predict $\hat{y}=1$ (has disease) only if a treatment is very expensive, invasive, or risky. We don't want to treat healthy people.
- **Action:** **Raise the threshold** (e.g., to 0.7 or 0.9).
- **Result:**
  - **Higher Precision:** The model only predicts `1` when it's highly confident, so it makes fewer False Positives.
  - **Lower Recall:** The model will be too cautious and miss more of the actual positive cases (it will have more **False Negatives, FN**).

#### **Scenario 2: Prioritizing High Recall (Being "Safe")**

- **Goal:** We want to find _all_ possible positive cases. We want to avoid **False Negatives (FN)**.
- **Example:** Predict $\hat{y}=1$ (has disease) if a disease is very dangerous if _not_ treated, but the treatment itself is cheap and safe. We'd rather treat 10 healthy people than miss one sick person.
- **Action:** **Lower the threshold** (e.g., to 0.3).
- **Result:**
  - **Higher Recall:** The model predicts `1` more readily, so it correctly identifies more of the actual positive cases.
  - **Lower Precision:** The model will also incorrectly label more negative cases as positive (it will have more **False Positives, FP**).

This relationship can be visualized on a **Precision-Recall Curve**, which shows the trade-off as you vary the threshold.

---

### **The F1-Score: Combining Metrics**

Since we now have two metrics, how do we compare models?

- Model A: High Precision, Low Recall
- Model B: Low Precision, High Recall

We need a single number that combines both. A simple **average** (`(P+R)/2`) is **not a good metric** because it can be "gamed" by a useless model (e.g., a model with 100% recall but 1% precision would look good).

The standard solution is the **F1-Score**.

- **What it is:** The **harmonic mean** of Precision and Recall.
- **Key Property:** The F1-Score gives more weight to the _lower_ of the two values. It will only be high if **both** precision and recall are high.
- **Formula:**
  $$F_1 = 2 \cdot \frac{Precision \times Recall}{Precision + Recall}$$

By calculating the F1-Score, you can get a single, more reliable metric to compare different algorithms or threshold choices for your skewed dataset.
