In PAC learning settings, to find the best hypothesis $h^*$ we need of *2 parameters* to indicate the goodness of the ERM rule

> $h^*$ is approximately good if $L_D (h_S) \leq \epsilon$  (**accuracy parameter $\epsilon$**) with probability $\geq 1 - \delta$ (**confidence level parameter $\delta$**)

In facts, with a finite set $\mathcal{H}<+\infty$ we can *almost always* (with probability $\geq 1 - \delta$) find an approximately good hypothesis ($L_D (h_S) \leq \epsilon$) if we have *enough data* in train set with 

$$
m \geq \frac{log(\frac{|\mathcal{H}|}{\delta})}{\epsilon}
$$

where $m=|S|$ (the proof of this theorem is in "quaderno").

So we can get a **definition of PAC learnability:**
$\mathcal{H}$ is *PAC learnable* if there exists $m_{\mathcal{H}}: (0,1)^2 \longrightarrow N$  and a learning algorithm $A(S)$ such that the realizability assumption holds with respect to $\mathcal{H}, D, f$, then when running $A(S)$  on 

$$
 m \geq m_{\mathcal{H}} (\delta, \epsilon)
 $$
 i.i.d examples generated by $D^m$ and labelled by $f$, the learning algorithm return $h\in \mathcal{H}$ with probability $\geq 1 - \delta$
  
$$
L_D (h_S) \leq \epsilon
$$

$m_{\mathcal{H}}$ is the *sample complexity* that represent how many examples are required to guarantee a PC solution

 
