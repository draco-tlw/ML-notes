### **Information Gain: How to Choose the Best Split 🌳**

When building a decision tree, the algorithm must decide which feature to use for a split at each node (e.g., split on "Ear Shape," "Face Shape," or "Whiskers").

The best split is the one that **reduces entropy the most**. This reduction in entropy is formally known as **information gain**. The algorithm's goal is to pick the feature that provides the **highest information gain**.

---

### **Calculating Information Gain**

To find the best feature, we calculate the information gain for _every possible split_ and pick the highest one.

The process has 3 steps:

1.  Calculate the entropy of the "child" nodes for each possible split.
2.  Calculate the weighted average entropy for each split.
3.  Calculate the information gain by subtracting the weighted average from the parent's entropy.

#### **Reference: Parent Node Entropy**

First, we need the entropy of the node _before_ the split (the "parent" node). In our root node example, we have 10 examples (5 cats, 5 dogs).

- $p_1 = 5/10 = 0.5$
- **Parent Entropy ($H_{root}$) = 1.0** (maximum impurity)

---

#### **Step 1: Calculate Entropy for Each Potential Split**

We'll test all three of our features:

- **Split 1: "Ear Shape"**
  - **Left Branch (Pointy):** 5 examples (4 cats, 1 dog). $p_1 = 4/5 = 0.8$.
    - _Entropy = 0.72_
  - **Right Branch (Floppy):** 5 examples (1 cat, 4 dogs). $p_1 = 1/5 = 0.2$.
    - _Entropy = 0.72_

- **Split 2: "Face Shape"**
  - **Left Branch (Round):** 7 examples (4 cats, 3 dogs). $p_1 = 4/7 \approx 0.57$.
    - _Entropy = 0.99_
  - **Right Branch (Not Round):** 3 examples (1 cat, 2 dogs). $p_1 = 1/3 \approx 0.33$.
    - _Entropy = 0.92_

- **Split 3: "Whiskers"**
  - **Left Branch (Present):** 4 examples (3 cats, 1 dog). $p_1 = 3/4 = 0.75$.
    - _Entropy = 0.81_
  - **Right Branch (Absent):** 6 examples (2 cats, 4 dogs). $p_1 = 2/6 \approx 0.33$.
    - _Entropy = 0.92_

---

#### **Step 2: Calculate the Weighted Average Entropy**

To get a single impurity score for each _split_, we calculate a **weighted average** of the child node entropies. The weight is the fraction of examples that went into that branch.

- **Ear Shape (Weighted Avg):**
  - $(\frac{5}{10} \times 0.72) + (\frac{5}{10} \times 0.72) = \mathbf{0.72}$

- **Face Shape (Weighted Avg):**
  - $(\frac{7}{10} \times 0.99) + (\frac{3}{10} \times 0.92) = 0.693 + 0.276 = \mathbf{0.969}$

- **Whiskers (Weighted Avg):**
  - $(\frac{4}{10} \times 0.81) + (\frac{6}{10} \times 0.92) = 0.324 + 0.552 = \mathbf{0.876}$

---

#### **Step 3: Calculate Information Gain (Select the Best Split)**

Information Gain = (Parent Entropy) - (Weighted Average Entropy of Children)

- **Ear Shape Gain:**
  - $1.0 - 0.72 = \mathbf{0.28}$

- **Face Shape Gain:**
  - $1.0 - 0.969 = \mathbf{0.03}$

- **Whiskers Gain:**
  - $1.0 - 0.876 = \mathbf{0.12}$

**Decision:** The "Ear Shape" split results in the **highest information gain** (0.28), so the algorithm chooses it as the feature for the root node.

_(This process is then repeated at each new node, using only the subset of data at that node.)_

---

### **Formal Definition of Information Gain 🧮**

- $H(p_1^{root})$ = Entropy of the parent node
- $w^{left}$ = Fraction of examples going to the left child
- $H(p_1^{left})$ = Entropy of the left child
- $w^{right}$ = Fraction of examples going to the right child
- $H(p_1^{right})$ = Entropy of the right child

$$Information Gain = H(p_1^{root}) - \left[ w^{left} H(p_1^{left}) + w^{right} H(p_1^{right}) \right]$$
