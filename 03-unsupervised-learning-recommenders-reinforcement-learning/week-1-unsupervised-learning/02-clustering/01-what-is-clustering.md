## **Introduction to Clustering**

**Clustering** is a type of unsupervised learning algorithm. Its goal is to look at a number of data points and automatically find points that are related or similar to each other, grouping them into "clusters."

### **Supervised vs. Unsupervised Learning**

To understand clustering, it helps to contrast it with the supervised learning methods (like logistic regression) you learned previously.

| Feature           | Supervised Learning                                           | Unsupervised Learning                                                  |
| :---------------- | :------------------------------------------------------------ | :--------------------------------------------------------------------- |
| **Data**          | Includes input features ($x$) and **target labels ($y$)**.    | Includes **only input features ($x$)**. No target labels are provided. |
| **Visualization** | Data points are distinct (e.g., $x$'s and $o$'s).             | Data points look the same (e.g., all dots).                            |
| **Goal**          | Find a decision boundary to predict the "right answer" ($y$). | Find "interesting structure" or patterns within the data.              |
| **Guidance**      | We tell the algorithm what the correct output is.             | We ask the algorithm to figure it out on its own.                      |

- **In the plot above:** You can see how an unsupervised algorithm looks at raw data (without labels) and identifies distinct groups (clusters) based on the proximity of the points to one another.

---

### **Real-World Applications of Clustering**

Clustering is widely used across many industries to find hidden structures in data:

- **News Aggregation:** Grouping similar news articles together (e.g., grouping all stories about "Pandas" or a specific event) to present them as a single topic.
- **Market Segmentation:** Analyzing a database of customers to discover distinct groups.
  - _Example:_ Deeplearning.ai discovered groups such as "Career Developers," "Skill Growers," and "AI Updaters" to better tailor content.
- **DNA Analysis:** Analyzing genetic expression data to group individuals who exhibit similar genetic traits.
- **Astronomy:** Analyzing astronomical data to group celestial bodies together to identify galaxies or coherent structures in space.
