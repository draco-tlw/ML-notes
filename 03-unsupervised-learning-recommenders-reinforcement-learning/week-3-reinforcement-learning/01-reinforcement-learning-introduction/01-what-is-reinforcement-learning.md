## **Introduction to Reinforcement Learning (RL) 🚁**

**Reinforcement Learning** is the third pillar of Machine Learning (alongside Supervised and Unsupervised Learning). While it currently has fewer commercial applications than supervised learning, it is incredibly powerful for tasks involving control and decision-making.

### **The Core Concept: Training a Dog 🐶**

The best way to understand RL is to think about training a puppy.

- You cannot explicitly tell a puppy "Move your left paw 3 inches forward."
- Instead, you watch the puppy behave.
- **Positive Reward:** If it sits, you say "Good Dog!" (+1 Reward).
- **Negative Reward:** If it chews the shoe, you say "Bad Dog!" (-100 Reward).
- Over time, the puppy learns to do more of the things that get rewards and fewer of the things that get punishments.

**Reinforcement Learning works the same way:** You don't tell the computer _how_ to do a task; you just tell it _what_ a good outcome looks like (the reward).

---

### **Example: The Autonomous Helicopter**

Imagine you want to program a helicopter to fly upside down (an autonomous stunt).

- **The Problem with Supervised Learning:**
  - To use Supervised Learning, you would need a dataset of $(x, y)$.
  - $x$ = Helicopter state (position, speed, orientation).
  - $y$ = The _perfect_ control stick movement.
  - **Issue:** It is extremely difficult (even for experts) to define exactly what the "perfect" stick movement is for every possible millisecond of flight.

- **The Reinforcement Learning Approach:**
  - **State ($s$):** The helicopter's current position and orientation.
  - **Action ($a$):** How to move the control sticks.
  - **Reward Function:**
    - Flying stable? **+1 Reward** (Good Helicopter!)
    - Crashed? **-1000 Reward** (Bad Helicopter!)
  - **Result:** The algorithm experiments and learns a function (policy) that maps states ($s$) to actions ($a$) to maximize the total reward.

---

### **Applications of Reinforcement Learning**

1.  **Robotics:**
    - Getting a robot dog to climb over complex obstacles.
    - Autonomous flight (helicopters, drones).
    - Landing a spacecraft (You will do this in the practice lab with a Lunar Lander!).
2.  **Game Playing:**
    - Chess, Go, Checkers.
    - Video games (e.g., teaching AI to play Mario or Dota).
3.  **Industrial & Financial Optimization:**
    - **Factory Optimization:** Maximizing throughput and efficiency.
    - **Stock Trading:** Determining the optimal strategy to sell large amounts of stock over time without crashing the price.

---

### **Summary**

- **Supervised Learning:** You tell the computer exactly what to do (State $x \rightarrow$ Correct Action $y$).
- **Reinforcement Learning:** You tell the computer the _goal_ via rewards (State $s \rightarrow$ Action $a \rightarrow$ Reward or Punishment). The computer figures out the "how" by itself.
