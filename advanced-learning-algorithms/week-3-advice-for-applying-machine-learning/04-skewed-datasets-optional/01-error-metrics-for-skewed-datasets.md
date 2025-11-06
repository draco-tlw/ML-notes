## **Skewed Datasets (Optional)**

A **skewed dataset** is a classification problem where the ratio of positive (1) to negative (0) examples is very far from 50/50.

- **Example:** Rare disease detection, where only 0.5% of patients have the disease ($y=1$) and 99.5% do not ($y=0$).

---

### **The Problem with Classification Error**

Using **classification error** (or **accuracy**) as your only metric is highly misleading for skewed datasets.

- **Example:** If the disease is present in only 0.5% of the population, a "dumb" (and useless) algorithm that _always_ predicts $y=0$ will have:
  - **0.5% error**
  - **99.5% accuracy**
- If your complex machine learning model achieves a 1% error (99% accuracy), it _looks_ good, but it's performing _worse_ than the useless algorithm that just guesses "no" every time.
- This shows that accuracy is not a good metric for telling if your model is actually useful.

---

### **Better Metrics: Precision and Recall**

For skewed datasets, we use two different metrics: **Precision** and **Recall**. To calculate them, we first build a **Confusion Matrix**.

#### **The Confusion Matrix**

A confusion matrix is a 2x2 table that breaks down your model's predictions on a test set (or CV set) into four categories:

|                        | **Actual Class: 1** (Has Disease)                     | **Actual Class: 0** (No Disease)                     |
| :--------------------- | :---------------------------------------------------- | :--------------------------------------------------- |
| **Predicted Class: 1** | **True Positive (TP)**<br>_Correctly predicted "yes"_ | **False Positive (FP)**<br>_Wrongly predicted "yes"_ |
| **Predicted Class: 0** | **False Negative (FN)**<br>_Wrongly predicted "no"_   | **True Negative (TN)**<br>_Correctly predicted "no"_ |

---

#### **Precision**

- **Question:** Of all the times the model predicted a patient **has the disease** ($y=1$), what fraction of those patients **actually** had the disease?
- **Intuition:** "When my model says 'yes', how often is it right?" (This is a measure of _trustworthiness_).
- **Formula:**
  $$Precision = \frac{\text{True Positives}}{\text{Total Predicted Positives}} = \frac{TP}{TP + FP}$$

#### **Recall**

- **Question:** Of all the patients who **actually had the disease**, what fraction did the model **correctly detect**?
- **Intuition:** "Of all the 'yes' cases that actually exist, how many did my model find?" (This is a measure of _completeness_).
- **Formula:**
  $$Recall = \frac{\text{True Positives}}{\text{Total Actual Positives}} = \frac{TP}{TP + FN}$$

---

### **Using Precision and Recall**

- The "dumb" algorithm that _always_ predicts $y=0$ would have 0 True Positives (TP).
- Its **Recall would be 0%** ($\frac{0}{0 + FN}$).
- This metric immediately and correctly tells you the algorithm is useless.
- A good algorithm must achieve **both** high precision and high recall.
