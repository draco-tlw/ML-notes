## **Building Intuition: Changing Rewards and Discount Factors 🎛️**

The best way to understand how Reinforcement Learning works is to see how the agent's behavior (Policy) changes when you tweak the "rules of the world." In the optional lab, we modify the **Mars Rover** example to see how $Q(s, a)$ adapts.

### **1. The Baseline Scenario**

- **Left Reward (State 1):** +100
- **Right Reward (State 6):** +40
- **Discount Factor ($\gamma$):** 0.5
- **Original Policy:** Go Left from states 2, 3, 4. Go Right from state 5.

---

### **Experiment 1: Changing the Reward**

**Scenario:** What if we drastically reduce the reward on the right side?

- **Change:** We lower the Right Reward from **+40** to **+10**.
- **Effect on $Q(s, a)$:**
  - Previously, at State 5, going Right was worth it (Reward 40 vs. traveling far for 100).
  - Now, going Right yields very little value. Even though the big reward (+100) is far away, it is mathematically worth the trip compared to the measly +10 nearby.
- **New Policy:** The robot now goes **Left** from _every_ state, including State 5.

---

### **Experiment 2: Increasing Patience ($\gamma$)**

**Scenario:** What if we make the robot more patient and forward-thinking?

- **Change:** We increase the Discount Factor ($\gamma$) from 0.5 to **0.9**.
- **Effect on $Q(s, a)$:**
  - $\gamma=0.9$ means future rewards retain most of their value (0.9, 0.81, 0.72...).
  - The robot is no longer "afraid" of the distance to the big reward.
  - At State 5, the "return" for traveling all the way left to the +100 reward becomes higher than grabbing the immediate +40.
- **New Policy:** The robot goes **Left** from State 5 (and all other states).

---

### **Experiment 3: Increasing Impatience ($\gamma$)**

**Scenario:** What if we make the robot extremely short-sighted?

- **Change:** We decrease the Discount Factor ($\gamma$) from 0.5 to **0.3**.
- **Effect on $Q(s, a)$:**
  - $\gamma=0.3$ means future rewards lose value incredibly fast (0.3, 0.09, 0.027...).
  - Ideally, the robot wants +100. But from State 4, the +100 is "too far away" in terms of discount steps. The value decays to almost nothing before the robot gets there.
  - The +40 on the right is closer.
- **New Policy:** The robot now goes **Right** from State 4 (whereas it used to go Left). It settles for the smaller reward because it is closer.

---

### **Summary of Intuition**

| Adjustment                     | Resulting Behavior                                                                                                                          |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **Lower a Reward**             | The agent will likely avoid that path and travel further to find a better reward.                                                           |
| **High $\gamma$ (e.g., 0.99)** | **Patient / Strategic.** The agent will ignore small nearby rewards to travel long distances for a massive payout.                          |
| **Low $\gamma$ (e.g., 0.1)**   | **Impatient / Greedy.** The agent cares only about what is right next to it. It will choose small nearby rewards over massive distant ones. |
