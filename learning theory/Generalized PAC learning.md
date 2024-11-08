The *realizability assumption*, as previously, states that there exists an hypothesis PAC in $\mathcal{H}$ that *perfectly label every instance point* for which $L_D(h)=0$, but this **but this is not always true**

So, since that we want to find the *optimal hypothesis* for which *minimizes the true error*, we can **relax the realizability assumption**:
instead supponsig $X\sim D$, we are going to assume that each pairs $(\vec{x}_i, y_i)$ are obtained by **a jointly distribution $Z=(X \times Y) \sim D$** where the label assigned on $\vec{x}_i$ is obtained in according to 

$$
D((\vec{x}, y) \ | \ \vec{x})= \mathbb{Pr}_{(\vec{x}, y)\sim D} [y \ | \ \ \vec{x}]
$$

**New type of the true error:** 

$$
		L_D(h)=\mathbb{Pr}_{(\vec{x}, y)\sim D} [h(\vec{x}) \neq y]=D(\{(\vec{x}, y) \ | \ h(\vec{x}) \neq y\})
$$

**New type of the empirical error:**

$$
L_S(h_S) = \frac{|\{i \quad | \quad h_s(\vec{x}_i) \neq y_i, \forall i=1\dots m\}|}{m} = \frac{1}{m} \sum_{i=1}^m 1(h(\vec{x}_i) \neq y_i)
$$

we prove that $\mathbb{E}[L_S (h)]=L_D$:

$$
\begin{align*}
\mathbb{E}[L_S(h)] &= \frac{1}{m} \sum_{i=1}^m \mathbb{E}[ \mathbf{1}(h(x_i) \neq y_i)] \\
&= \frac{1}{m} \sum_{i=1}^m \mathbb{Pr}_{(x_i, y_i) \sim \mathcal{D}}[h(x_i) \neq y_i] \\
&= \frac{1}{m} \sum_{i=1}^m L_{\mathcal{D}}(h) \\
&= \frac{1}{m} (m L_{\mathcal{D}}(h)) = L_{\mathcal{D}}(h).
\end{align*}
$$

in estimator theory, we just to proof that $L_S (h)$ is *a correct estimator* for $L_D (h)$.

**The new ERM rule above 0-1 classifier:** Since that we want to find the optimal predictor $h^* \in \mathcal{}h$ that minimize the train error, for a 0-1 classifier there exist the *bayes optimal predictor* given by

$$
f_D(\mathbf{x}) =
\begin{cases}
1, & \text{se } P_D(y = 1 \mid \mathbf{x}) \geq 0.5 \\
0, & \text{otherwise}.
\end{cases}
$$
it follows from this a prop.

For any classifier $g:X \longrightarrow \{0,1\}$ it holds that $L_D(f_D) \leq L_D (g)$ 

Now we can get a **definition of agnostic PAC learnability:**
$\mathcal{H}$ is *agnostic PAC learnable* if there exists $m_{\mathcal{H}}: (0,1)^2 \longrightarrow N$  and a learning algorithm $A(S)$ such that the realizability assumption holds with respect to $\mathcal{H}, D, f$, then when running $A(S)$  on 

$$
 m \geq m_{\mathcal{H}} (\delta, \epsilon)
 $$
 i.i.d examples generated by $D^m$ and labelled by $f$, the learning algorithm return $h\in \mathcal{H}$ with probability $\geq 1 - \delta$
  
$$
L_{\mathcal{D}}(h) \leq \min_{h' \in \mathcal{H}} L_{\mathcal{D}}(h') + \varepsilon
$$

where $m_{\mathcal{H}}$ is the *sample complexity* that represent how many examples are required to guarantee a PC solution while 

$$
L_D (h) = \mathbb{E}_{(\vec{x}, y)\sim D} [\ell((\vec{x}, y), h)]
$$
