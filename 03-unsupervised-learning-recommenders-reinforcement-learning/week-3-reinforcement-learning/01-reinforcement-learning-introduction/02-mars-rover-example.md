## **The Mars Rover Example 🤖**

We use a simplified example inspired by the Mars Rover to formalize the reinforcement learning problem.

### **The Setup**

- **States ($s$):** There are 6 possible positions (boxes) for the rover.
  - $s_1, s_2, s_3, s_4, s_5, s_6$.
  - The rover starts at a specific state (e.g., $s_4$).
- **Rewards ($R(s)$):** Each state has an associated reward value.
  - $s_1$: **+100** (Highly interesting science target).
  - $s_6$: **+40** (Moderately interesting science target).
  - $s_2, s_3, s_4, s_5$: **0** (No significant science value).
- **Terminal States:** Once the rover reaches state 1 or state 6, the "day" ends. It collects the reward and stops moving.
- **Actions ($a$):** At each step, the rover can move:
  - **Left** ($\leftarrow$)
  - **Right** ($\to$)

---

### **The Reinforcement Learning Loop**

At every time step, the following sequence occurs:

1.  **Current State ($s$):** The robot is in a specific position (e.g., state 4).
2.  **Reward ($R(s)$):** The robot receives the reward associated with the _current_ state (e.g., 0).
3.  **Action ($a$):** The robot chooses an action (e.g., "Go Left").
4.  **Next State ($s'$):** As a result of the action, the robot transitions to a new state (e.g., state 3).

This tuple **$(s, a, R(s), s')$**—Current State, Action, Reward, Next State—is the fundamental building block of RL algorithms.

---

### **Example Trajectories**

- **Option A (Go Left):**
  - Start at 4 ($R=0$) $\rightarrow$ Go Left $\rightarrow$ At 3 ($R=0$) $\rightarrow$ Go Left $\rightarrow$ At 2 ($R=0$) $\rightarrow$ Go Left $\rightarrow$ At 1 ($R=100$).
  - **Total Reward:** 100. (Plus it reached the best target).

- **Option B (Go Right):**
  - Start at 4 ($R=0$) $\rightarrow$ Go Right $\rightarrow$ At 5 ($R=0$) $\rightarrow$ Go Right $\rightarrow$ At 6 ($R=40$).
  - **Total Reward:** 40. (Reached a target, but a less valuable one).

- **Option C (Wasted Motion):**
  - Start at 4 $\rightarrow$ Go Right (to 5) $\rightarrow$ Go Left (back to 4) $\rightarrow$ Go Left (to 3) ... $\rightarrow$ At 1 ($R=100$).
  - **Result:** Still gets +100 reward eventually, but wasted time and fuel.

The goal of the algorithm is to figure out that **Option A** is the best strategy.
