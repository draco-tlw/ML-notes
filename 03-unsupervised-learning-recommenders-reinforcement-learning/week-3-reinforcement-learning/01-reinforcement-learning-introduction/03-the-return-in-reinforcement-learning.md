## **The Return in Reinforcement Learning 💰**

In Reinforcement Learning, the goal isn't just to get a single reward, but to maximize the total sum of rewards over time. However, getting a reward _now_ is usually better than getting the same reward _later_.

### **The Analogy: $5 Now vs. $10 Later**

- Imagine a $5 bill is at your feet. You can pick it up instantly.
- Imagine a $10 bill is across town (30-minute walk).
- Which do you choose?
  - The $10 is larger, but requires effort and time.
  - If the effort/delay is too high, the $5 bill becomes more attractive.
- **The Concept:** Rewards obtained quickly are more valuable than rewards that take a long time to achieve.

---

### **Defining the Return**

The **Return** is the sum of future rewards, but weighted by a **Discount Factor ($\gamma$, Gamma)**.

- **Formula:**
  If the sequence of rewards is $R_1, R_2, R_3, \dots$
  $$\text{Return} = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \dots$$

- **The Discount Factor ($\gamma$):**
  - A number between 0 and 1 (usually close to 1, like 0.9 or 0.99).
  - It determines how "impatient" the algorithm is.
  - **$\gamma^0 = 1$:** The first reward gets 100% credit.
  - **$\gamma^1 = 0.9$:** The second reward gets 90% credit.
  - **$\gamma^2 = 0.81$:** The third reward gets 81% credit.

---

### **Example Calculation (Mars Rover)**

Assume $\gamma = 0.5$ (very impatient).
Start at State 4.
Goal: Reach State 1 (Reward 100).
Path: $4 \rightarrow 3 \rightarrow 2 \rightarrow 1$.

- **Step 1 (State 4):** Reward = 0
- **Step 2 (State 3):** Reward = 0
- **Step 3 (State 2):** Reward = 0
- **Step 4 (State 1):** Reward = 100 (Terminal)

**Calculate Return:**
$$\text{Return} = 0 + (0.5 \times 0) + (0.5^2 \times 0) + (0.5^3 \times 100)$$
$$\text{Return} = 0 + 0 + 0 + (0.125 \times 100) = \mathbf{12.5}$$

- Even though the final reward is 100, because it is 3 steps away and $\gamma=0.5$, its _present value_ (Return) is only 12.5.

---

### **Comparing Policies**

The return depends heavily on the **actions** the robot chooses.

- **Scenario A: Always Go Left** (Toward the 100 reward)
  - Start at State 4 $\rightarrow$ Return = **12.5**
  - Start at State 5 $\rightarrow$ Return = **6.25** (Farther away)

- **Scenario B: Always Go Right** (Toward the 40 reward)
  - Start at State 4 $\rightarrow$ Return = **10**
  - Start at State 5 $\rightarrow$ Return = **20**

- **Scenario C: Mixed Strategy**
  - If at State 4, go Left (Return 12.5 > 10).
  - If at State 5, go Right (Return 20 > 6.25).
  - **Insight:** The optimal strategy changes based on your current state to maximize the return.

---

### **Negative Rewards**

What if rewards are negative (punishments)?

- The discount factor incentivizes the agent to **delay** the punishment as long as possible.
- _Analogy:_ Paying a $10 fine today hurts more than paying a $10 fine 5 years from now (due to inflation/interest). The algorithm tries to push the negative event into the distant future.
