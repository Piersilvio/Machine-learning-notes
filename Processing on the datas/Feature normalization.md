Feature normalization is a technique for *transforming input variables* (features) *so that they have a common scale*. This is often **useful** or **necessary** in many machine learning algorithms, **where variables with different scales** can negatively affect the performance of the model.

> Any *normalized data* is such if the **average** of each feature across the train set **is 0** and the **standard deviation** $\sigma$  of each feature **is 1** (*Standard deviation*)

Given 

$$

X = \begin{bmatrix} 
x_1^{(1)} & x_2^{(1)} & \dots & x_n^{(1)} \\ 
x_1^{(2)} & x_2^{(2)} & \dots & x_n^{(2)} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
x_1^{(m)} & x_2^{(m)} & \dots & x_n^{(m)} 
\end{bmatrix} \in \mathbb{R}^{m \times n},
\quad
\vec{x}^{(i)} = \begin{bmatrix} 
x_1^{(i)} \\ 
x_2^{(i)} \\ 
\vdots \\ 
x_n^{(i)} 
\end{bmatrix} \in \mathbb{R}^n,
\quad
\vec{y} = \begin{bmatrix} 
y^{(1)} \\ 
y^{(2)} \\ 
\vdots \\ 
y^{(m)} 
\end{bmatrix} \in \mathbb{R}^m
$$

before to combine $X$ and $\vec{y}$, the *norm.* phase must be applied both structure.

**Min-max scaler:** scales each feature to a predefined range, often to $[-1,1]$ or $[1,0]$. Fore each feature $j$ , the normalized value $\hat{x}_j^{(i)}$ is given by

$$
\hat{x}_j^{(i)} = \frac{x_j^{(i)}-min(X_j)}{max(X_j) - min(X_j)}
$$

**Normal score normalization:** *standardization* transforms features so that they have a mean of *zero* and a standard deviation of *one*.

$$
\hat{x}_j^{(i)} = \frac{x_j^{(i)}-\mu_j}{\sigma_j}
$$

where

$$
\mu_j = \frac{1}{m} \sum_{i=1}^m x_j^{(i)} 
$$
$$
\sigma_j = \sqrt{\frac{1}{m} \sum_{i=1}^m (x_j^{(i)} - \mu_j)^2}
$$

without normalization, large values ​​in the train set would dominate the generalization.