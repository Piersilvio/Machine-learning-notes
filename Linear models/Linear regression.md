The hypothesis class concides with the class of [[Linear predictors]]. A linear regression solve a regression task considering a a $d$-dimensional domain set where $Y=\mathbb{R}$ 

$$
\mathcal{H}_{reg} = \{\vec{x} \longrightarrow \langle \vec{w}, \vec{x}\rangle \ | \ \vec{w} \in \mathbb{R}^{d+1}\} 
$$
$$
h_{\vec{w}} (\vec{x}) = w_0 + w_1 x_1 + w_2 x_2 \dots + w_d x_d
$$

**Loss function for generalization error:** squared loss

$$
L_D(h_{\vec{w}}(\vec{x})) = (h_{\vec{w}}(\vec{x}) - y)^2
$$

**Loss function for empirical error:** Mean squared error (MSE)

$$
L_s\left(h_{\vec{w}}(\vec{x})\right)=\frac{1}{m} \sum_{i=1}^m\left(h_{\vec{w}}\left(\vec{x}_i\right)-y_i\right), \quad m=|S|
$$

To *estimate* $\vec{w}$ a **common learning algorithm** involves using the **least squares method**: the goal is is to minimize the *sum of the squared errors* (the difference between the observed and predicted values) solving a linear system in *matrix form*

Let's consider a design matrix

$$
X=\left[\begin{array}{c}
\vec{x}_1 \\
\vec{x}_2 \\
\vdots \\
\vec{x}_m
\end{array}\right] \in R^{m \times(d+1)}, \quad \vec{y}=\left[\begin{array}{c}
y_1 \\
\vdots \\
y_m
\end{array}\right] \in R^m, \quad \vec{w}=\left[\begin{array}{c}
\vec{w}_i \\
\vdots \\
\vec{w}_{d+1}
\end{array}\right] \in R^{(d+1) \times 1}
$$

the **ERM rule** in this case, is given considering the *residual square error* (RSS)

$$
\begin{gathered}
\underset{\vec{w}}{\operatorname{argmin}} L_S\left(h_{\vec{w}}\right)=\underset{\vec{w}}{\operatorname{argmin}} \frac{1}{m} \sum_{i=1}^m\left(\left\langle\vec{w}, \vec{x}_i\right\rangle-y_i\right)^2 \\
\Leftrightarrow \text { equivalent form } \\
\underset{\vec{w}}{\operatorname{argmin}} \sum_{i=1}^m\left(\left\langle\vec{w}, \vec{x}_i\right\rangle-y_i\right)^2=\underset{\vec{w}}{\operatorname{argmin}} \operatorname{RSS}(\vec{w}) \\
\text { (Residual Square Error) }
\end{gathered}
$$

using the design matrix and solving the square...

$$
\underset{\vec{w}}{\operatorname{argmin}} \operatorname{RSS}(\vec{w})=\underset{\vec{w}}{\operatorname{argmin}}(\vec{y}-X \vec{w})^T(\vec{y}-X \vec{w})
$$

to find the optimal $\vec{w}$, we can compute the *gradient* of $(\vec{y}-X \vec{w})^T(\vec{y}-X \vec{w})$ and compare to 0 to solve in rispect of $\vec{w}$ 

$$
\begin{gathered}
\frac{\partial \operatorname{RSS}(\vec{w})}{\partial \vec{w}}=0 \\
\Leftrightarrow \quad-2 X^T(\vec{y}-X \vec{w})=0 \\
\Leftrightarrow \quad-2 X^T \vec{y}+2 X^T X \vec{w}=0 \\
\Leftrightarrow \quad X^T X \vec{w}=X^T \vec{y} \\
\Leftrightarrow \quad\left(X^T X\right)^{-1} X^T X \vec{w}=\left(X^T X\right)^{-1} X^T \vec{y} \quad\left(\mathrm{X}^T \mathrm{X} \text { is invertible }\right) \\
\Leftrightarrow \quad \vec{w}=\left(X^T X\right)^{-1} X^T \vec{y} \quad \text { The optimal } \vec{w}^*
\end{gathered}
$$

If $X^T X$ is *not invertible*, let $A=X^T X$ and $A^+$ the generalized inverse of $A$ i.e. $AA^+ A = A$ (*pseudoinverse of rose-penrose*). Then

$$
    w=A^+ X^T \vec{y} \quad \text{is a optimal solution for } X^T X \vec{w} = X^T \vec{y}
$$

In this case, to compute $A^+$ we can note that $A=X^T X$  is *symmetric*, so we can use the *eigenvalue composition* of $A$ to find $A^+$ . The form of an eigenvalue composition is given by 

$$
A^+=VD^+ V^T
$$

where $V$ is a orthonormal matrix for which $V^T V = I\in \mathbb{R}^{d \times d}$ and $D_{i,j}^+$ is a diagonal matrix. Let's prove that $A^+$ is a generalized inverse of $A$ (i.e. $AA^+ A = A$ )

$$
AA^+ A= (VDV^T)(VD^+ V^T)(VDV^T)=VDD^+ DV^T = VDV^T = A
$$

**Other learning algorithms:** makes use of *regularization* and *Stocastic gradient discendt* 