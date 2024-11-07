A **polinomial regression** is a particular case of [[Linear regression]] that make use non-linear transformation (in particular polynomials of degree $r$)to model non-linear relationship between $X$ and $Y$. 

Suppose we have included the bias term 

$$
\begin{array}{r}
\vec{w}=\left(b, w_1, w_2, \ldots, w_d\right)^T \in R^{d+1} \\
\vec{x}=\left(1, x_1, x_2, \ldots, x_d\right) \in R^{d+1}
\end{array}
$$

Hypothesis class of polynomials degree $r$ in $X=\mathbb{R}$ and $Y=\mathbb{R}$:

$$
\mathcal{H}_{pol}^r = \{\vec{x} \longrightarrow p_{\vec{w}}(\vec{x}) \quad | \quad \vec{x} \in X\}
$$
$$
h_{\vec{w}}(\vec{x})=p_{\vec{w}}(\vec{x})=w_0 + w_1 x + w_2 x^2 + w_3 x^3 \dots +w_r x^r = \langle \vec{w}, \vec{x} \rangle
$$

where the grade $r$ represent the complexity of the trained model

![[Pasted image 20241106102110.png]]

**Pre-processing on instances:** Since the hypothesis is a polynomial (and therefore *non-linear with respect to $\vec{x}$* ), to train this model we need to pre-process the $\vec{x}$ features making the *hypothesis function linear in the training parameters*

> **I reduce a polynomial problem to a linear problem with the [[Feature expansions of data]]**

to do this, we use the a **feature polynomial expansion of $\vec{x}$** using a mapping function $\phi(\vec{x})$  that produce a new instance's vector $\vec{x}'\in \mathbb{R}^{d+1}$. Consider $X=\mathbb{R}^d$  

$$
\vec{x}' = \left[ 1, x_1, x_2, \dots, x_d, x_1^2, x_2^2, \dots, x_d^2, \dots, x_1^r, x_2^r, \dots, x_d^r, x_1 x_2, x_1 x_3, \dots, x_{d-1} x_d \right]
$$

in this way, the definition of hypothesis class *changes* and, after that, we can train the model with the least square in [[Linear regression]] model method to find the optimal $\vec{w}^*$ .

$$
\mathcal{H}_{pol}^r = \{\vec{x} \longrightarrow p_{\vec{w}}(\phi (\vec{x})) \quad | \quad \vec{x} \in X\}
$$
$$
h_{\vec{w}}(\phi (\vec{x}))=p_{\vec{w}}(\phi (\vec{x}))=w_0 + w_1 x + w_2 x^2 + w_3 x^3 \dots +w_r x^r = \langle \vec{w}, \vec{x}' \rangle
$$
