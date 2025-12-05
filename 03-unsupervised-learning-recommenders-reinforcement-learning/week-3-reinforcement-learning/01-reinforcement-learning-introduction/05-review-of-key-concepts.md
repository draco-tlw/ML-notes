## **Review: Key Concepts of Reinforcement Learning 📝**

We have established a standardized formalism to describe any Reinforcement Learning problem, regardless of whether it is a simple Mars Rover or a complex autonomous helicopter.

### **The Core Components**

| Concept             |  Symbol  | Definition                                                                                                           |
| :------------------ | :------: | :------------------------------------------------------------------------------------------------------------------- |
| **State**           |   $s$    | The current situation or position of the agent (e.g., Rover is in box 4).                                            |
| **Action**          |   $a$    | What the agent decides to do (e.g., Move Left).                                                                      |
| **Reward**          |  $R(s)$  | The immediate feedback received for being in a state (e.g., +100 for finding science, 0 for empty rock).             |
| **Discount Factor** | $\gamma$ | A number between 0 and 1 (e.g., 0.99). It determines how much we value future rewards compared to immediate rewards. |
| **Return**          |          | The total sum of discounted future rewards. The goal is to maximize this.                                            |
| **Policy**          | $\pi(s)$ | The function (or "brain") that maps a state to an action. It tells the agent _what to do_ in any given situation.    |

---

## **The Markov Decision Process (MDP) 🧠**

This entire formalism—States, Actions, Rewards, and the transition logic—is collectively called a **Markov Decision Process (MDP)**.

### **The "Markov" Property**

The term "Markov" implies a specific property about the system:

- **The future depends _only_ on the current state.**
- It does **not** depend on the history of how you got there.
- _Example:_ In chess, to decide the next best move, you only need to look at the current board. It does not matter _how_ the pieces arrived at those positions; the history is irrelevant to the decision-making process.

### **The MDP Cycle**

The MDP represents a continuous loop of interaction:

1.  **Agent** observes the current state ($s$).
2.  **Agent** uses its Policy ($\pi$) to choose an Action ($a$).
3.  **Environment** reacts to the action.
4.  **Environment** returns a new State ($s'$) and a Reward ($R$) to the Agent.
5.  Repeat.

---

## **Applications of the Formalism**

The exact same mathematical framework applies to vastly different real-world problems.

| Application    | State ($s$)                            | Action ($a$)    | Reward ($R$)                 |
| :------------- | :------------------------------------- | :-------------- | :--------------------------- |
| **Mars Rover** | Position (1-6)                         | Left, Right     | +100 (Science), 0 (Nothing)  |
| **Helicopter** | Position, Orientation, Speed, Rotation | Stick movements | +1 (Stable), -1000 (Crash)   |
| **Chess**      | Position of all pieces on the board    | Legal moves     | +1 (Win), -1 (Loss), 0 (Tie) |

**Note on Discount Factors:**

- **Rover Example:** We used $\gamma = 0.5$ (very impatient) for illustration.
- **Real World:** We usually use $\gamma \approx 0.99$ or $0.999$ (patient), as we care about the long-term outcome (winning the game or keeping the helicopter flying).
