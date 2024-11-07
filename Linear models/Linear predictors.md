A **linear predictor** is a math. model which use a linear combination of indipendent variable to make prediction about a dipendent variable. 

![[Linear_regression.svg#invert_B]]

**Ingredients:**  we need of 
- $\vec{w}=(w_0, \dots, w_d)$ a **weight's vector** that, assigned to each instances point, determines the *slope* of the linear function that best fit the data; 
- a $b \in \mathbb{R}$ **bias term** that allows to better fit the linear function translating the line on the plane in order to better learn the weights for the line.

**Hypothesis class of affine functions:** In $X=\mathbb{R}^d$ 

$$
L_d=\left\{\vec{x} \longrightarrow\langle\vec{w}, \vec{x}\rangle+b \mid \vec{w} \in R^d, b \in R\right\}
$$

where 

$$
h_{\vec{w}, b}(\vec{x})=\left\langle\vec{w}, \overrightarrow{x_i}\right\rangle+b=\left(\sum_{i=1}^d w_i x_i\right)+b
$$

If we include the bias term into $\vec{w}$,  so considering $X=\mathbb{R}^{d+1}$ 

$$
\begin{array}{r}
\vec{w}=\left(b, w_1, w_2, \ldots, w_d\right)^T \in R^{d+1} \\
\vec{x}=\left(1, x_1, x_2, \ldots, x_d\right) \in R^{d+1}
\end{array}
$$

we have the following notation to semplify the calculus

$$
h_{\mathbf{w}}(\mathbf{x})=\langle\mathbf{w}, \mathbf{x}\rangle+b=\langle\mathbf{w}, \mathbf{x}\rangle
$$
