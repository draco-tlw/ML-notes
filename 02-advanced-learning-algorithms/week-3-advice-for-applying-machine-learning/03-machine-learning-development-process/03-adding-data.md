### **The Machine Learning Development Process**

Developing a machine learning system involves iterating on both the **model (code)** and the **data**.

- **Model-Centric Approach:** The traditional research approach, where you hold the **data fixed** and iterate on improving the **model/algorithm**.
- **Data-Centric Approach:** A modern, practical approach where you often hold the **model fixed** and focus on improving the **quality and quantity of the data**. This is often a more efficient way to improve performance.

---

### **Methods for Adding Data**

Simply "getting more data" of everything can be slow and expensive. These techniques offer more targeted and efficient ways to improve your dataset.

#### 1. Targeted Data Collection

Instead of gathering all data, use **error analysis** to guide your efforts.

- **Example:** If your spam classifier's biggest problem is "pharma spam," a targeted effort to find and label _only_ more pharma spam emails will be far more efficient and effective than just adding more random emails to your dataset.

#### 2. Data Augmentation 🔄

This technique creates _new_ training examples by applying realistic distortions to your _existing_ training examples. The label for the new, distorted example remains the same.

- **Goal:** Tell the algorithm what types of variations are "allowed" without changing the core identity of the data.
- **Key Principle:** The distortions you apply should be **representative** of the types of noise or variations you expect to see in your test set. (e.g., adding random pixel noise to an image is _not_ helpful if that's not a realistic scenario).

- **Examples for Images (OCR):**
  - Rotate the image slightly.
  - Enlarge or shrink the image.
  - Change the contrast.
  - Apply random "warping" to the image grid.

- **Examples for Audio (Speech Recognition):**
  - Take the original audio ("What is today's weather?").
  - Add background noise (e.g., a crowd, a car).
  - Simulate a bad cell phone connection.

#### 3. Data Synthesis

This is the process of creating _brand new, artificial_ data from scratch.

- **Goal:** To generate a massive amount of realistic-looking training data.
- **Example (Photo OCR - Optical Character Recognition):**
  - **Real Data:** Pictures of text in the real world.
  - **Synthetic Data:** Use your computer to:
    1.  Get thousands of different fonts.
    2.  Type random characters and words.
    3.  Apply random colors, backgrounds, rotations, and distortions.
  - This can generate a nearly infinite supply of training examples that look very similar to real-world text.
- **Use Cases:** Data synthesis is most common and effective in **computer vision** tasks.

---

When you can't easily get or create more data, a technique called **transfer learning** can be a powerful alternative.
