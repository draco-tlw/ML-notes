## Face Recognition

### Introduction to Face Recognition

**Overview and Applications**
Face recognition technology is one of the premier special applications of convolutional neural networks (CNNs). Real-world implementations, such as the security access systems used at corporate headquarters, allow seamless entry by identifying users in real time without physical ID cards. These architectures typically integrate multiple computer vision components to maintain high security against spoofing attacks.

**Face Verification vs. Face Recognition**
It is important to distinguish between two primary tasks in computer vision literature:

- **Face Verification:** A $1:1$ matching problem where the system is given an input image along with a claimed identity (name or ID). The objective is to verify whether the input image matches the claimed identity.
- **Face Recognition:** A much more challenging $1:K$ problem. Given an input image, the system must check it against a database of $K$ people to determine the specific identity of the person.

**Mathematical Complexity of Recognition**
The face recognition task is significantly harder than face verification because the probability of making an error scales with the size of the database. Suppose a face verification system has an accuracy of $99\%$, representing an individual error rate of $\epsilon = 0.01$. If this system is applied directly to a recognition task with a database size of $K = 100$ people, the system has $100$ times more opportunities to make a mistake. Assuming independent checks, the overall probability of making at least one incorrect match across a database of $K$ individuals can be conceptually modeled as:

$$P(\text{error}) = 1 - (1 - \epsilon)^K$$

To maintain an acceptable recognition error across a large database of $K$ individuals, the underlying verification system must achieve a much higher baseline accuracy, such as $99.9\%$ or above. Consequently, highly optimized face verification architectures are used as the fundamental building blocks for robust face recognition systems.

**Liveness Detection**
A critical component of a secure face recognition deployment is ensuring that the target is a live human rather than a photograph. This is known as liveness detection, and it prevents spoofing attempts (e.g., using a printed photo on an ID card to bypass authentication). Liveness detection can be treated as a binary supervised learning problem trained to predict whether an input belongs to a live human versus a non-live object.

**One-Shot Learning Challenge**
One of the primary difficulties in building a practical face verification system is the requirement to solve a one-shot learning problem. In typical real-world applications, the system must be capable of accurately verifying or recognizing a person after being given only $1$ training sample of their face.

Key Terms/Formulas:

- **Face Verification:** A $1:1$ matching system that determines if an input image belongs to a specific, claimed identity.
- **Face Recognition:** A $1:K$ system that identifies an input person out of a database containing $K$ known individuals.
- **Liveness Detection:** A supervised learning classifier used to distinguish a live human from a spoofing medium (e.g., a static image).
- **One-Shot Learning:** A learning paradigm where a model must correctly identify a class or individual given only $1$ training example.

---

### One-Shot Learning for Face Recognition

**The One-Shot Learning Challenge**
In most real-world face recognition applications, the system must be capable of identifying a person given only a single training example or template image. Historically, deep learning architectures struggle to achieve high performance when restricted to an extremely small dataset. For instance, an employee database might only contain one picture per team member. When an individual approaches an office turnstile, the system must correctly evaluate their identity against these single baseline reference images or detect that they do not belong to the database.

**Failure of Standard Softmax Architectures**
A naive approach to this problem would involve passing an input image through a standard Convolutional Network (ConvNet) that feeds into a softmax unit. The softmax layer could output a classification label $y$ spanning a discrete set of classes, such as the number of employees plus an additional class for "none of the above". However, this approach fails dramatically in practice for two main reasons:

- **Insufficient Data:** A training set with only one image per person provides far too few gradients to learn robust, generalized visual features natively through a standard classification pipeline.
- **Dynamic Database Scale:** If a new employee joins the organization, the database changes size. This forces the addition of a new output node to the softmax layer, changing the classification dimension from $K$ outputs to $K+1$ outputs. Consequently, the entire ConvNet would have to be retrained from scratch every time the database changes, which is computationally intractable for practical deployments.

**The Similarity Function Approach**
To bypass the limitations of classification architectures, face recognition systems learn a universal similarity function rather than predicting static categorical classes. The system trains a specialized network to evaluate a distance metric, denoted as $d$, which takes a pair of images as inputs and computes the precise degree of visual difference between them.

The structural objective of this function can be formulated as follows:

$$d(\text{image}^{(1)}, \text{image}^{(2)}) = \text{degree of difference}$$

If the two input images depict the same individual, the network is trained to output a small numerical value. Conversely, if the inputs depict two entirely different individuals, the network outputs a significantly larger value.

**Verification and Recognition Workflows**
During deployment, the verification problem is resolved by comparing the query image against the claimed identity template image using the distance function $d$. The binary decision boundary is governed by a hyperparameter threshold denoted as $\tau$:

- If $d(\text{image}^{(1)}, \text{image}^{(2)}) \leq \tau$, the system predicts that the two images represent the same person.
- If $d(\text{image}^{(1)}, \text{image}^{(2)}) > \tau$, the system predicts that the images depict different individuals.

To extend this mechanism to a multi-class face recognition task, the new query image is sequentially evaluated against every single image stored in the corporate database using the function $d$. For example, if a query image matches an employee in the database, the pairwise function will yield a value well below $\tau$ for that specific match, while yielding values significantly higher than $\tau$ for all other references. If an unregistered individual attempts entry, all pairwise computations will safely exceed the threshold $\tau$, prompting the system to correctly declare that the individual is not found within the database. This eliminates the need for network retraining when adding new individuals; adding a member simply requires appending a new reference image to the database.

Key Terms/Formulas:

- **One-Shot Learning:** A learning paradigm where an algorithm must learn to accurately recognize a class or identity from only a single training instance.
- **Similarity Function ($d$):** A learned function that inputs a pair of images and outputs a scalar representing their degree of difference, where a lower value implies identical ownership.
- **Threshold ($\tau$):** A hyperparameter boundary used to binary-classify whether a computed difference $d$ indicates a matching identity ($d \leq \tau$) or a non-matching identity ($d > \tau$).
- **Label ($y$):** The output target or prediction variable representing the system's final classification decision.

---

### Siamese Networks for Face Recognition

**From Classification to Encodings**
Standard convolutional neural network pipelines typically pass an input image through a series of convolutional, pooling, and fully connected layers before feeding the final vector into a softmax unit for multi-class classification. For face recognition, however, the softmax layer is discarded. Instead, the focus shifts to a dense layer deeper within the network that outputs a feature vector containing a fixed number of dimensions, such as a $128$-dimensional vector. This vector is represented as $f(x_i)$, which serves as a highly compressed, descriptive encoding of the raw input image $x_i$.

**Siamese Network Architecture**
To calculate the similarity between two distinct face images, denoted as $x_i$ and $x_j$, both images are evaluated using a unique framework known as a Siamese neural network. In this setup, both inputs are fed into two separate branches of the exact same convolutional neural network. Crucially, these twin networks share the exact same parameters and architectural weights.

Once the network computes the respective representations $f(x_i)$ and $f(x_j)$ for both images, the system defines a universal distance function $d$ to mathematically evaluate their degree of difference. This distance is calculated as the norm of the difference between their two encodings:

$$d(x_i, x_j) = \|f(x_i) - f(x_j)\|$$

This methodology of evaluating identical, parallel networks with shared parameters was a foundational element of the _DeepFace_ system, an influential research project developed by Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, and Lior Wolf.

**Training and Optimization Goals**
The core objective when training a Siamese network is to optimize the shared internal parameters using backpropagation so that the resulting feature encodings capture true facial similarity. The network parameters are updated according to two primary conditions:

- If two images $x_i$ and $x_j$ belong to the **same person**, the network must adjust its weights so that the distance between their encodings is small: $\|f(x_i) - f(x_j)\| \to 0$.
- If two images $x_i$ and $x_j$ belong to **different people**, the network must adjust its weights so that the distance between their encodings is large: $\|f(x_i) - f(x_j)\| \gg 0$.

By altering the parameters across the convolutional and fully connected layers via gradient descent, the model learns to map raw facial pixels into a vector space where identical identities sit close together and different identities are pushed far apart. This prepares the network for optimization via structured objective functions, such as the triplet loss function.

Key Terms/Formulas:

- **Siamese Neural Network:** A deep learning architecture that feeds multiple inputs into identical subnetworks sharing the same weights to compare their extracted feature vectors.
- **Encoding ($f(x_i)$):** A vectorized representation of an input image $x_i$ generated by a deep layer in the neural network, translating visual attributes into a dense numeric space.
- **Distance Function ($d(x_i, x_j)$):** The mathematical norm of the difference between two computed encodings, determining how similar or different two input faces are:
  $$d(x_i, x_j) = \|f(x_i) - f(x_j)\|$$

- **DeepFace:** A milestone face recognition system that demonstrated the effectiveness of utilizing deep neural network encodings and Siamese-style comparisons for identity verification.

---

### The Triplet Loss Function

**Concept of Triplets**
To learn the parameters of a neural network that produces effective face encodings, models utilize gradient descent optimized via the triplet loss function. This technique requires the network to simultaneously evaluate three distinct images at a time, creating a "triplet."

- **Anchor ($A$):** The baseline reference image of an individual.
- **Positive ($P$):** A different image of the exact same individual as the anchor.
- **Negative ($N$):** An image of a completely different individual.

**Mathematical Formulation and the Margin**
The objective of the network is to minimize the distance between the anchor and positive image encodings while maximizing the distance between the anchor and negative image encodings. Formally, we want the distance metric to satisfy the property:

$$d(A, P) \leq d(A, N)$$

Expressed as a squared norm of the differences between the network representations, the target state is:

$$\|f(A) - f(P)\|^2 \leq \|f(A) - f(N)\|^2$$

Moving all terms to the left-hand side yields:

$$\|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 \leq 0$$

However, minimizing this equation directly allows the neural network to learn trivial solutions. For example, if the network maps every single input image to a vector of all zeros ($f(x) = \vec{0}$) or outputs an identical constant vector for all faces, the difference expression resolves trivially to $0 - 0 = 0$, satisfying the inequality without learning meaningful facial features. To prevent this behavior, a hyperparameter called the margin, denoted as $\alpha$, is introduced to ensure the network forces a distinct gap between pairs. The formulation is modified to:

$$\|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha \leq 0$$

If the margin $\alpha$ is set to $0.2$ and the anchor-positive distance $\|f(A) - f(P)\|^2$ is $0.5$, an anchor-negative distance of $0.51$ is insufficient despite being larger. The network demands that $\|f(A) - f(N)\|^2$ be at least $0.7$ or greater, pushing identical identities closer together and differing identities further apart.

**The Triplet Loss Objective**
The loss function $\mathcal{L}$ computed for a single triplet of images $(A, P, N)$ is defined by selecting the maximum value between zero and the margin-adjusted distance expression:

$$\mathcal{L}(A, P, N) = \max\left(\|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha, 0\right)$$

Taking the maximum ensures that if the neural network successfully satisfies the condition—making the green-lighted expression less than or equal to zero—the loss for that triplet evaluates to exactly $0$. If the condition is violated and the value exceeds zero, it yields a positive loss value that gradient descent will minimize. The overall cost function $J$ for the entire neural network is the summation of individual losses across all available triplets in the training set containing $m$ examples:

$$J = \sum_{i=1}^{m} \mathcal{L}(A^{(i)}, P^{(i)}, N^{(i)})$$

**Training Requirements and Hard Triplet Mining**
To construct a training set for triplet loss, the source dataset must contain multiple images of the same person. For example, a training dataset might contain $10,000$ total pictures composed of $1,000$ unique individuals, averaging $10$ pictures per person. If a dataset only contains one picture per person, it is impossible to form anchor-positive pairs, meaning the system cannot be trained (even though a fully-trained model can evaluate one-shot learning problems during deployment).

A critical consideration in training is how triplets are selected. If triplets are chosen completely at random, the distance constraint is too easy to satisfy because two randomly selected individuals will naturally look completely different, meaning $\|f(A) - f(N)\|^2$ will already be significantly larger than $\|f(A) - f(P)\|^2 + \alpha$. In this scenario, the loss evaluates to $0$ almost immediately, providing no useful gradients for the network to update its weights.

To achieve optimal computational efficiency and distinct representations, models utilize "hard triplet mining," as detailed in the _FaceNet_ paper by Florian Schroff, Dmitry Kalenichenko, and James Philbin. Hard triplets deliberately select combinations where the anchor-positive distance is very close to or smaller than the anchor-negative distance:

$$d(A, P) \approx d(A, N)$$

This forces the gradient descent algorithm to perform heavy optimization work to push the parameters further apart, maximizing learning efficiency.

**Large-Scale Commercial Datasets and Pre-trained Models**
Modern commercial face recognition systems require immense data scale, commonly training on datasets ranging from $1$ million to upwards of $100$ million images. Because gathering datasets of this scale is remarkably difficult for standard implementations, a common industry practice is to download highly robust, pre-trained model weights (such as FaceNet or DeepFace architectures) available online rather than executing the entire optimization procedure from scratch.

Key Terms/Formulas:

- **Anchor ($A$):** The primary reference face image in a triplet.
- **Positive ($P$):** An image representing the exact same identity as the anchor image.
- **Negative ($N$):** An image representing a completely different identity than the anchor image.
- **Margin ($\alpha$):** A positive hyperparameter that enforces a minimal mathematical gap between the anchor-positive and anchor-negative pairs, preventing the network from settling on uniform or zero-vector trivial solutions.
- **Triplet Loss Function ($\mathcal{L}$):** The objective formula defined as $\max\left(\|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha, 0\right)$ which evaluates the error of a triplet.
- **Total Cost ($J$):** The cumulative sum of all individual triplet losses across the training dataset:
  $$J = \sum_{i=1}^{m} \mathcal{L}(A^{(i)}, P^{(i)}, N^{(i)})$$

- **Hard Triplets:** Specifically curated triplets where $d(A, P) \approx d(A, N)$, which maximizes backpropagation efficiency by forcing the model to fix near-incorrect or difficult encodings.

---

### Face Recognition as a Binary Classification Problem

**Binary Classification Formulation**
As an alternative to utilizing the triplet loss function, face verification and recognition can be framed entirely as a standard binary classification problem. In this paradigm, a Siamese network architecture is employed where twin convolutional neural networks share identical parameters and weights. Instead of computing distances between triplets, the system processes a pair of images, $x^{(i)}$ and $x^{(j)}$, passing each through the shared network to generate dense feature vector embeddings. These embeddings, typically spanning a $128$-dimensional space or higher, are subsequently fed into a logistic regression unit that acts as the final classifier. The training network is optimized using backpropagation over these image pairs.

- The target output label $y$ is set to $1$ when the pair consists of images depicting the exact same individual.
- The target output label $y$ is set to $0$ when the images depict two entirely different individuals.

**Mathematical Architecture**
The final logistic regression unit does not evaluate the raw embeddings directly. Instead, it computes a set of input features based on the element-wise absolute differences between the components of the two vector representations. The predicted output $\hat{y}$ is determined by applying the sigmoid function $\sigma$ to a linear combination of these difference features:

$$\hat{y} = \sigma \left( \sum_{k=1}^{128} w_k |f(x^{(i)})_k - f(x^{(j)})_k| + b \right)$$

In this equation, $f(x^{(i)})_k$ and $f(x^{(j)})_k$ represent the $k$-th component of the respective $128$-dimensional face encodings for images $x^{(i)}$ and $x^{(j)}$. The model learns weights denoted by $w_k$ along with a bias parameter $b$ during the training phase. This allows the network to automatically determine which specific facial features are most critical when predicting whether two faces match.

**Alternative Similarity Metrics**
There are several mathematical variations available for constructing the difference features fed into the logistic regression equation. An alternative formulation explored in the pioneering _DeepFace_ paper replaces the element-wise absolute difference with a variant known as the chi-square ($\chi^2$) similarity metric. The core component of this variation modifies the feature calculation to evaluate the normalized squared difference:

$$\frac{(f(x^{(i)})_k - f(x^{(j)})_k)^2}{f(x^{(i)})_k + f(x^{(j)})_k}$$

When integrated into the full predictive classification model, the formula expands to include the learned weights $w_k$ and bias $b$:

$$\hat{y} = \sigma \left( \sum_{k=1}^{128} w_k \frac{(f(x^{(i)})_k - f(x^{(j)})_k)^2}{f(x^{(i)})_k + f(x^{(j)})_k} + b \right)$$

This specific metric offers an effective alternative configuration depending on the exact distribution of face data and alignment techniques.

**Computational Deployment Optimization**
Deploying an enterprise-grade face recognition system utilizing this Siamese architecture requires maximizing computational efficiency, especially when managing a massive database of employee identities. A crucial engineering trick involves pre-computing the feature embeddings for all known individuals in the database ahead of time.

- When a person walks up to a security turnstile or doorway, the network only needs to execute a single forward pass through the upper network branch to generate the encoding for that new live image.
- The system then takes this freshly computed encoding and directly evaluates it against the pre-computed database encodings using the logistic regression formula to obtain $\hat{y}$.
- This optimization avoids storing raw, uncompressed images in memory and prevents the highly redundant recalculation of database embeddings every single time an identity check occurs. This technique is highly effective for both binary classification architectures and triplet loss frameworks.

Key Terms/Formulas:

- **Binary Classification:** A supervised learning paradigm where a model predicts one of two discrete classes, such as $y \in \{0, 1\}$, to determine identity matches.
- **Element-Wise Absolute Difference Feature:** An input feature extracted by computing $|f(x^{(i)})_k - f(x^{(j)})_k|$ across all dimensions $k$ of two parallel face embeddings.
- **Chi-Square ($\chi^2$) Similarity:** An alternative similarity component defined as $\frac{(f(x^{(i)})_k - f(x^{(j)})_k)^2}{f(x^{(i)})_k + f(x^{(j)})_k}$ used to weigh normalized differences between facial encodings.
- **Pre-computation Optimization:** A computational design pattern where template feature encodings are calculated once and stored in storage, minimizing real-time forward pass operations during live network deployment.
- **Prediction Output ($\hat{y}$):** The final probability scalar generated by the sigmoid activation function representing the likelihood that two face images belong to the same person.
