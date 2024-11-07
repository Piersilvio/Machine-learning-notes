Feature polynomial expansion is a *feature transformation* technique that consists of generating new variables (features) by calculating polynomial powers and combinations of the original features.

> this helps the learning model on representing non-linear relationship between $X$ and $Y$ with a linear model that makes use a learning algorithm

Formally, given a vector $\vec{x}=(x_1, \dots, x_d)$ the **feature polynomial expansion** is given by

$$
\vec{x}' = \left[ 1, x_1, x_2, \dots, x_d, x_1^2, x_2^2, \dots, x_d^2, \dots, x_1^r, x_2^r, \dots, x_d^r, x_1 x_2, x_1 x_3, \dots, x_{d-1} x_d \right]
$$

---

Example: 

![[Pasted image 20241106111931.png]]


---

However, By *increasing* the degree $r$ and the size of the original features $d$, the number of new features *grows rapidly*, which can lead to a significant increase in computational complexity and the **risk of overfitting**.

> To solve this problem, we want to find a trade-off choosing the best model that fit well the data, and to do this, we use the [[Model selection and validation]]

