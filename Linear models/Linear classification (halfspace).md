We are going to define the **hypothesis class of halfspace**. This type of class make use of the class of [[Linear predictors]] which to return in output the **sign of a linear function** refers to a class of linear functions that **divides** the feature space into 2 parts with a linear function used as the **boundary**.

Let's assume $X=\mathbb{R}^{d+1}$ and $Y=\{-1,1\}$ and, including the bias term in the weight's vector, the hypothesis class is given by

$$
\mathcal{H}_{HS} = \{\vec{x} \longrightarrow sign(h_{\vec{w}}) \ \ | \ \ h_{\vec{w}} \in L_d\}
$$

the hypothesis class takes an instance $\vec{x} \in X$ and returns the *sign of the linear function* described above, parametrized by $\vec{w}$.

![[img9.png]]

as the bias varies, halfspace can translate up and down, while as the weight vector varies, the slope of the line that best fits the data changes.

The **ERM goal** is to find the best configuration of $\vec{w}$ that separate well the data. By def. we know that 

$$
\begin{aligned}
h_{\vec{w}}(\vec{x_i}) = \langle \vec{w}, \vec{x_i} \rangle > 0 &\implies h_{\vec{w}}(\vec{x_i}) = +1 = y_i; \\
h_{\vec{w}}(\vec{x_i}) = \langle \vec{w}, \vec{x_i} \rangle < 0 &\implies h_{\vec{w}}(\vec{x_i}) = -1 = y_i;
\end{aligned}
$$

combining this rules in one formula, we have that $y_i h_{\vec{w}}(\vec{x_i})=\langle \vec{w}, \vec{x_i}\rangle y_i$ where

$$
\begin{align}
    y_i = +1 \ \land \ h_{\vec{w}}(\vec{x_i}) > 0 \quad \implies \quad \langle \vec{w}, \vec{x_i} \rangle y_i > 0 \\
y_i = -1 \ \land \ h_{\vec{w}}(\vec{x_i}) < 0 \quad \implies \quad \langle \vec{w}, \vec{x_i} \rangle y_i > 0
\end{align}
$$

where the correct classification both part is given by the **linear separability** of the data, defined by a arbitrary weight's vector $\vec{w}$. 

**Realizability assumption:** requires that the data *must be linearly separable* if there exists a weight's vector $\vec{w}$ such that $\langle \vec{w}, \vec{x_i} \rangle y_i > 0$ for all training set poins. 

## Learning algorithm: perceptron

The first method that achieve this goal is the **perceptron**. This algorithm takes in *input* a sequence of pair $(\vec{x_1}, y_1) \dots (\vec{x_m}, y_m)$ from a training set $S$ and *initialize* a zeros vector $\vec{w}^{(1)} = (0, \dots , 0)$. At the *iterate* $t$ if there *is a missclassified points in $S$*, then update the weight's vector $\vec{w}^{(t+1)}$ the iterate

$$
\vec{w}^{(t+1)} \longleftarrow \vec{w}^{(t)} + y_i \vec{x}_i
$$

pseudocode:

```
Input: (x_1, y_1), ..., (x_m, y_m)
initialize w_0 = (0, ..., 0)

for each t = 1, 2, ... do
	if exist i = 1,..., m such that y_i <w_i, x_i> <= 0 
	then 
		w_(t+1) = w_t + y_i x_i
	else 
		return w_t
```

the method termination is guaranteed i.f.f. the data are linearly separable.

The perceptron is *easy* to implement and it used for *only* linearly separable data: in this case the termination is guaranteed and it *required an exponential number of iteration in* $d$, with a *several potentially solutions*, which one is picked depend on starting values. 

However, *it does not guarantee a good solution for non-linearly separable data* as the
updating of the weights is not based on the optimization of a loss function.

## Learning algorithm: Gradient discendt and Stocastic version

**Gradient descent** make use (unlike the standard implementation already seen) of a generalized loss function (we have the ERM hyp. now!) which, optimized based on the weight vector, guarantees its minimization to find the best hypothesis.

**Gradient of a function:** 

$$
\nabla f(\vec{w}) = \left( 
\frac{\partial f(\mathbf{w})}{\partial w_1}, \dots, \frac{\partial f(\mathbf{w})}{\partial w_d}
\right)
$$

The main idea below GD involves moving along the direction of the negative gradient $\nabla f(\vec{w})$, i.e. descending in the direction of max. descendt.

**Loss function considered:** RELU 

$$
\ell(h_{\vec{w}}, (\vec{x}, y))= \max \{0, -y\langle \vec{w}, \vec{x} \rangle \}
$$

![[chapter_deep-learning-basics_mlp_3_0.svg]]

In our case, assume that we want to minimizing (i.e. in according to ERM) the train loss function

$$
 L_S(h_{\vec{w}})=\frac{1}{m}\sum_{i=1}^m \ell(\vec{w}, (\vec{x}_i, y_i)) =f(\vec{w})
$$

As mentioned before,  GD takes in *input* a sequence of pair $(\vec{x_1}, y_1) \dots (\vec{x_m}, y_m)$ from a training set $S$ and *initialize* a zeros vector $\vec{w}^{(0)} = (0, \dots , 0)$ and at the *iterate* $t$ update the weight's vector $\vec{w}^{(t+1)}$ the iterate

$$
\vec{w}^{(t+1)} \longleftarrow \vec{w}^{(t)} - \eta \nabla f(\vec{w})
$$

and return the *mean vector* 

$$
\hat{w} = \frac{1}{T} \sum_{t=1}^T \vec{w}^{(t)}
$$

pseudocode:

![[Pasted image 20241105160054.png]]

About the *learning rate* wher with high η means rapid descendt but the probability to skip and go further the min while low η means a very slow descendt and sure convergence.

**SGD**, unlike the GD, **don't required the update direction to be exactly on the gradient** $\nabla f(\vec{x})$. Instead, we allow the direction to be a **random vector** and only require that its the **expected value** at each step will equal to the gradient direction.

![[Pasted image 20241105160430.png]]

**Computation of the random vector:** In SGD algorithm, we can note it choose a vector $\vec{v}^{(t)}$ at random from distribution such that $\mathbb{E}[\vec{v}^{(t)}|\vec{w}^{t}]\in \nabla L_S(h_{\vec{w}})$  (or $L_S(\vec{w})$).  Let's compute the value of $\vec{v}^{(t)}$. Take $i$ uniformly at random from $\{1,...,m\}$ and let $(\vec{x}',y')=(\vec{x}_i,y_i)$ be the corrisponding point in the training set; consider the vector $\nabla l(\vec{w}, (\vec{x}', y'))$. Now, we show that 

$$
\mathbb{E}[\nabla l(\vec{w}, (\vec{x}', y'))] = \nabla L_S(\vec{w})
$$

Note that GD consider the gradient of $L_S(\vec{w})$ (i.e., we are trying to minimize $L_S(\vec{w})$)

$$
\nabla L_S(\vec{w})=\nabla (\frac{1}{m}\sum_{i=1}^m l(\vec{w}, (\vec{x}_i,y_i ))) =  \frac{1}{m}\sum_{i=1}^m \nabla l(\vec{w}, (\vec{x}_i,y_i )))
$$

For SGD, instead, we have 

$$
\begin{align}
    \mathbb{E}[\nabla l(\vec{w}, (\vec{x}', y'))] = \sum_{i=1}^m \mathbb{Pr}[(\vec{x}', y')=(\vec{x}_i, y_i)] \cdot \nabla l(\vec{w}, (\vec{x}_i, y_i))=\frac{1}{m} \sum_{i=1}^m \nabla l(\vec{w}, (\vec{x}_i, y_i))=\nabla L_S(\vec{w})
\end{align}
$$

So, the general iterate becomes of SGD algorithm becomes

$$
\vec{w}^{(t+1)} \longleftarrow \vec{w}^{(t)}-\eta \nabla l(\vec{w}^{(t)}, (\vec{x}_i, y_i))
$$

Now, we are going to compute $l(\vec{w}^{(t)}, (\vec{x}_i, y_i))$. From our assumption, the loss that we have choose is given by the RELU $l(\vec{w}, (\vec{x}_i, y_i))= \max \{0, -y\langle \vec{w}, \vec{x}_i \rangle \}$. Appliyng the gradient on this loss,  we have that

$$
\nabla l (\vec{w}, (\vec{x_i}, y_i)) = 
\begin{cases} 
\vec{0} & \text{if } y_i \langle \vec{w}, \vec{x_i} \rangle > 0 \\
\nabla (-y_i \langle \vec{w}, \vec{x_i} \rangle) & \text{otherwise}
\end{cases}
$$

Where the otherwise case indicates all those $x_i$ that are misclassified by $\vec{w}$. Assume that $y_i\langle \vec{w}, \vec{x}_i \rangle < 0$ and let's compute this with gradient 

$$
\begin{align}
    \nabla (-y_i, \langle \vec{w}, \vec{x}_i \rangle)=[\frac{\partial (-y_i, \langle \vec{w}, \vec{x}_i \rangle)}{\partial w_1}, \dots, \frac{\partial (-y_i, \langle \vec{w}, \vec{x}_i \rangle)}{\partial w_d}]
\end{align}
$$

let $\vec{x}_ i =(x_{i,1}, \dots, x_{i,d})$, since that $-y_i\langle \vec{w}, \vec{x}_i\rangle = -y_i \cdot \sum_{j=1}^d w_j, x_{i,j}$ then

$$
\begin{align}
    \frac{\partial (-y_i, \langle \vec{w}, \vec{x}_i \rangle)}{\partial w_j}=-y_i x_{i,j}
\end{align}
$$

and this, implies that the all substitution inside the gradient vector of the loss

$$
\nabla l (\vec{w}, (\vec{x_i}, y_i)) = [-y_i x_{i,1}, \dots , -y_i x_{i,d}]^T
    =-y_i[x_{i, 1}, \dots , x_{i,d}]^T=-y_i \vec{x}_i
$$

In the for *cycle of SGD algorithm*, now we can set inside it the following condition

$$
\begin{align}
    \text{if } y_i \langle \vec{w}^{(t)}, \vec{x}_i \rangle < 0 \text{ then} \quad
    \left\{
    \begin{aligned}
        \vec{w}^{(t+1)} &\longleftarrow \vec{w}^{(t)} + \eta \ \  y_i\vec{x}_i \\
    \end{aligned}
    \right\}.
\end{align}
$$

that **coincides exactly** with the general iteration of the perceptron, with the only exception that in the perceptron $\eta = 1$, while in the SGD $\eta$ is a regularized parameter.