Let $A(S)$ a learning algorithm applies on a given train set $S$ that return $h_S$. 

**Main goal of training:** find the best $h_S$ which minimizes the true error $L_D(h_S)$. However, *we don't know* the real value for each instance in the train set.

**Empirical risk:** A possible solution to evaluate the performance of $A(S)$ consist on compute a *different loss directly on train set*

$$
L_S(h_S) = \frac{|\{i \quad | \quad h_s(\vec{x}_i) \neq y_i, \forall i=1\dots m\}|}{m}
$$

**ERM Goal:** in this way, during the training phase, we want to find the best $h_S$ which *minimizes the empirical risk* (or train error)

---

![[Pasted image 20241107120836.png]]

---

If we see the previous example, we can observe how a model found with only the ERM hypothesis **fails to generalize** the data well, and that it can learn by heart (in *Overfitting*): this because we have good result during the training ($L_S$) but, at the same time, we have poor gen. error ($L_D$).

**Solution:** let's apply ERM rule over a *restricted set of hypothesis $\mathcal{H}$* 

$$
\text{ERM}_{\mathcal{H}}(S) \in \operatorname{argmin}_{h \in \mathcal{H}} \{ L_S(h) \}
$$

that is the model $h\in \mathcal{H}$ picked from hyp. class that's minimizes the train error

**Inductive bias:** Since that $\mathcal{H}$ can be an *infinite set* we induce a bias in terms of dimension of $\mathcal{H}$ assuming that $\mathcal{H}<+\infty$

Now, to learn $h$ from ERM, we need of *two assumption*: 
the first is the **realizability assumption:** there exists best $h^* \in \mathcal{H}$ such that $L_D(h^*)$. Since that the ERM hypothesis is based on a given train set, this assumption implies that $L_S(h^*)$ 

the second assumption is the **i.i.d. assumption** where each examples in $S\sim D^m$ are *indipendently and identically distribuited*: above this assumption, a *found ERM hypothesis* can only be [[Probably and Approximately Correct (PAC)]].







