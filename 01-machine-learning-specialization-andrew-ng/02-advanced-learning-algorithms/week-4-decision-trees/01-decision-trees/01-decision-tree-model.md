## **Welcome to Week 4: Decision Trees**

This week introduces **decision trees** and **tree ensembles** (collections of trees). These algorithms are very powerful, widely used in many practical applications, and are a common tool for winning machine learning competitions.

---

### **Case Study: Cat Classification 🐱**

We will use a binary classification problem to understand decision trees: identifying if an animal is a cat or not (e.g., for an adoption center).

- **Problem:** Binary classification (Is it a cat? Yes/No).
- **Target (Y):** 1 (Cat) or 0 (Not Cat).
- **Features (X):** The features are **categorical**, meaning they take on one of a few discrete values.
  - **Ear Shape:** Pointy or Floppy
  - **Face Shape:** Round or Not Round
  - **Whiskers:** Present or Absent

---

### **What is a Decision Tree?**

A decision tree is a model that makes predictions by following a tree-like structure of "if-then" questions. The model _is_ the tree itself.

#### **How to Make a Prediction (Inference)**

You start at the top of the tree and follow the branches down based on your input example's features.

1.  Start at the **root node** (the topmost node).
2.  Read the feature question (e.g., "What is the ear shape?").
3.  Follow the branch that matches your example's value (e.g., the "Pointy" branch).
4.  Arrive at the next node (which could be another decision node or a leaf node).
5.  If it's a **decision node**, repeat the process (e.g., "What is the face shape?").
6.  If it's a **leaf node** (a box at the bottom), you stop. This node contains the final prediction (e.g., "Cat").

---

### **🌲 Key Terminology**

- **Node:** Any point in the tree (an oval or a rectangle).
- **Root Node:** The single, topmost node where you begin.
- **Decision Node:** An intermediate node that asks a question about a feature. It always splits into further branches.
- **Leaf Node:** A terminal node at the bottom of the tree. It does not ask questions but provides the final prediction.
  - _(Note: In computer science, trees are often drawn "upside down" like a hanging plant, with the root at the top and the leaves at the bottom.)_

---

### **The Learning Problem**

For any given dataset, there are many (perhaps thousands) of different decision trees that could be constructed.

The job of the **decision tree learning algorithm** is to automatically look at the training set and decide:

1.  Which features to ask questions about.
2.  In what order to ask them.
3.  What the final predictions should be at the leaves.

The goal is to build a tree that not only does well on the training set but also **generalizes** well to new, unseen examples.
