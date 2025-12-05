## **Stochastic Reinforcement Learning 🎲**

Up until now, we have assumed that the environment is **Deterministic**: if you command the robot to go Left, it _always_ goes Left.
In the real world, outcomes are rarely 100% reliable. Floors might be slippery, wind might blow a drone off course, or wheels might slip. We call these **Stochastic Environments** (environments involving randomness).

### **The Mars Rover Example (Stochastic Version)**

Imagine the same Mars Rover setup, but now the robot is imperfect.

- **Command:** Go Left.
  - **90% Chance ($0.9$):** Robot successfully moves **Left**.
  - **10% Chance ($0.1$):** Robot slips and accidentally moves **Right**.
- **Command:** Go Right.
  - **90% Chance ($0.9$):** Robot successfully moves **Right**.
  - **10% Chance ($0.1$):** Robot slips and accidentally moves **Left**.

### **The Challenge: Random Rewards**

Because the movement is random, the sequence of states and rewards is no longer guaranteed. Even if you follow the exact same Policy, the outcome can change every time you run the robot.

- **Run 1 (Lucky):** You command Left. It goes Left (Reward 0) $\rightarrow$ Left (Reward 0) $\rightarrow$ Left (Reward 100). **High Return.**
- **Run 2 (Unlucky):** You command Left. It slips Right (Reward 0). You command Left. It slips Right again (Reward 0). It takes much longer to reach the goal. **Low Return.**

---

### **The New Goal: Expected Return**

Since we cannot guarantee a specific return, we change our goal. We want to maximize the **Expected Return** (which is the statistical term for the **Average Return**).

If we ran the robot through the maze 1,000 times, we want the _average_ sum of discounted rewards to be as high as possible.

**Mathematical Notation:**
$$\text{Maximize } E[R_1 + \gamma R_2 + \gamma^2 R_3 + \dots]$$

- **$E[\dots]$**: The Expected Value (Average) operator.

---

### **Modified Bellman Equation**

The Bellman Equation must be updated to account for this randomness. Since the Next State ($s'$) is not guaranteed, we must take the **average** of the possible future outcomes.

$$Q(s, a) = R(s) + \gamma E[\max_{a'} Q(s', a')]$$

- **$R(s)$**: The immediate reward (this is usually known).
- **$E[\dots]$**: We average the best possible future returns ($Q(s', a')$) based on the probability of landing in each possible next state $s'$.

### **Intuition on Control**

- **Low Stochasticity (e.g., 1% slip):** You have high control. Your Q-values and Expected Returns will be high.
- **High Stochasticity (e.g., 40% slip):** You have low control. Your actions matter less because the environment is chaotic. Your Q-values and Expected Returns will decrease significantly.
