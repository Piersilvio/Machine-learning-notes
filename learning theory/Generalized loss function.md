Given any $\mathcal{H}$, a **loss function** is any function $\mathcal{l}:\mathcal{H} \times (X,Y) \longrightarrow \mathbb{R}_+$.

**True risk:** 

$$
L_D (h) = \mathbb{E}_{(\vec{x}, y)\sim D} [\ell((\vec{x}, y), h)]
$$

**Empirical risk:**

$$
L_S (h) = \frac{1}{m} \sum_{i=1}^m \ell((\vec{x}
_i, y_i), h)
$$

**0-1 loss:** 

$$
\ell_{0-1}(h, (x, y)) \triangleq \begin{cases} 
      0 & \text{if } h(x) = y \\
      1 & \text{if } h(x) \neq y 
   \end{cases}
$$

**Squared loss for regression:**

$$
\ell_{\text{sq}}(h, (x, y)) \triangleq \left( h(x) - y \right)^2

$$
