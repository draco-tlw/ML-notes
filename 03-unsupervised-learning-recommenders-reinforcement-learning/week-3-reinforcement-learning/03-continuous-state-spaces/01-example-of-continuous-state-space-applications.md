## **Continuous State Spaces in RL 🚘**

So far, we have looked at simple examples like the Mars Rover where the robot is in one of 6 specific "boxes" ($s_1, \dots, s_6$). This is a **Discrete State Space**.

In the real world, robots (cars, helicopters, drones) don't jump between discrete boxes. They move smoothly through space. Their position is defined by continuous numbers (like 2.5 meters, 10.43 degrees). This is a **Continuous State Space**.

### **Example 1: The Self-Driving Truck**

Imagine controlling a toy truck. To know its state, you need more than a single integer. You need a vector of continuous values:

- **Position:** $x, y$ coordinates.
- **Orientation:** $\theta$ (Angle it is facing, 0-360 degrees).
- **Velocity:** $\dot{x}, \dot{y}$ (Speed in x and y directions).
- **Angular Velocity:** $\dot{\theta}$ (How fast it is turning).

**State Vector ($s$):**
$$s = [x, y, \theta, \dot{x}, \dot{y}, \dot{\theta}]$$
This state is a vector of **6 numbers**.

---

### **Example 2: The Autonomous Helicopter**

A helicopter moves in 3D space, making the state even more complex.

- **Position:** $x, y, z$ (North/South, East/West, Altitude).
- **Orientation (Angles):**
  - **Roll ($\phi$):** Tilting Left/Right.
  - **Pitch ($\theta$):** Tilting Forward/Backward.
  - **Yaw ($\omega$):** Compass direction (North, East, etc.).
- **Velocities:** $\dot{x}, \dot{y}, \dot{z}$ (Speed in 3 directions).
- **Angular Velocities:** $\dot{\phi}, \dot{\theta}, \dot{\omega}$ (Rate of rotation).

**State Vector ($s$):**
$$s = [x, y, z, \phi, \theta, \omega, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\omega}]$$
This state is a vector of **12 continuous numbers**.

---

### **The Challenge**

In a Discrete State Space (like the 6-state rover), we could just create a table for the Q-function:

- Row 1: Q(State 1, Left)
- ...
- Row 6: Q(State 6, Right)

In a **Continuous State Space**, the state $s$ is a list of real numbers. Since there are infinite possible combinations of coordinates and speeds, **we cannot use a table**. We need a function that can take these continuous numbers as input and output the Q-value.
