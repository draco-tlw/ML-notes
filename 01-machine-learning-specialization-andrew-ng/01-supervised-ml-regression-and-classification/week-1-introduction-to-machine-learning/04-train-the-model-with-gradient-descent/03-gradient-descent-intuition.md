## **Gradient Descent Intuition**

To understand how gradient descent works, we can analyze the two key components of its update rule: the **derivative** and the **learning rate**.

### **The Role of the Derivative (`∂J/∂w`)**

Let's use a simplified cost function $J(w)$ with only one parameter, `w`, to visualize the process.

- The **derivative** at any point on the cost function curve is the **slope of the tangent line** at that point.
- The slope tells us which direction is "downhill" and helps the algorithm decide whether to increase or decrease the parameter `w`.

There are two main scenarios:

#### Case 1: The slope is positive

1. This happens when our current value of `w` is to the right of the minimum. The tangent line points up and to the right.
2. The update rule is: $w := w - \alpha \cdot (\text{positive number})$.
3. Since $\alpha$ is also positive, we are subtracting a positive value from `w`.
4. This causes `w` to **decrease**, moving it to the left and closer to the minimum.

#### Case 2: The slope is negative

1. This happens when our current value of `w` is to the left of the minimum. The tangent line points down and to the right.
2. The update rule is: $w := w - \alpha \cdot (\text{negative number})$.
3. Subtracting a negative number is the same as adding a positive number.
4. This causes `w` to **increase**, moving it to the right and closer to the minimum.

In both cases, the derivative term automatically guides the parameter update in the correct direction to lower the cost.
