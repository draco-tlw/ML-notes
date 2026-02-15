## **The Policy ($\pi$) 🧠**

In Reinforcement Learning, we need a formal way to define _how_ an agent chooses an action based on its current situation. This function is called the **Policy**.

### **Definition**

A Policy, denoted by the Greek letter **Pi ($\pi$)**, is a function that maps a **State ($s$)** to an **Action ($a$)**.

$$a = \pi(s)$$

- **Input:** The current State $s$.
- **Output:** The Action $a$ that the agent should take.

### **The Goal of Reinforcement Learning**

The ultimate goal of any RL algorithm is to find the **Optimal Policy** (often denoted as $\pi^*$).

- We are not just looking for _any_ policy.
- We want to find the specific policy that, when followed, results in the **maximum possible Return** (sum of discounted rewards).

### **Example: Mars Rover Policy**

Imagine a policy designed to grab the nearest reward:

| Current State ($s$) | Policy Output $\pi(s)$ | Meaning                         |
| :------------------ | :--------------------- | :------------------------------ |
| **State 2**         | Left                   | "If I am in State 2, go Left."  |
| **State 3**         | Left                   | "If I am in State 3, go Left."  |
| **State 4**         | Left                   | "If I am in State 4, go Left."  |
| **State 5**         | Right                  | "If I am in State 5, go Right." |

In this example, the policy $\pi$ is simply a lookup table or a set of rules telling the robot exactly what to do in every possible location.

### **Terminology Note**

- In control theory or robotics, this might be called a **"Controller."**
- In Reinforcement Learning, the standard term is always **"Policy."**
