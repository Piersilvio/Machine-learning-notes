A learner has access to

**Domain set $X$:** it’s the domain that contains all the possible instances that learner uses to make predictions. Usually $X$ is a $n$-dimensional space where each instance $\vec{x}_i$ contains values of the features.

**Label set $Y$:** for each $i$-instance in the domain set, we provides a label $y_i \in Y$, where $Y4 is the set of assumable value for $y_i$ (in classification task $Y=\{-1,1\}$ or $Y=\{0,1\}$)

**Data generation model:** *unknow to the learner*
1. each instance of domain set are generate by some *probability distribution* over $X$
2. there exist a labelling function s.t. $f(\vec{x}_i) =y_i$ 

**Training set $S$:** All point labeled in according to $f$, are contains in a *dataset* denoted *training set* which contains a sequence of pair $(\vec{x}_i, y_i)$ for all $i = 1 \dots m$ 

The **scope of a learner** is to return a function called *hypothesis* (or classifier/regressor) of type $h:X \longrightarrow Y$ that, *pre-trained on training data*, allows to *make prediction about new data*

$$
h(\vec{x}_i) =y_i \quad h \approx f \quad \forall i = 1 \dots m \quad (m=|S|)
$$

$h$ must be similar to $f$ and it provides by a learning algorithm on a given dataset $A(S)$ that produce $h_S$ 

**Measure of success:** an *error of a classifier* is given by the probability that the prediction is incorrect respect the real label assigned by f

$$
L_D(h) = \mathbb{Pr}_{X \sim D} \quad [h(\vec{x}) \neq f(\vec{x})] = D(\{\vec{x} \quad | \quad h(\vec{x}) \neq f(\vec{x})\})
$$

$L_D$ is the generalization error (o true loss) on the true label function $f$ and on the probability distribution $d$ and is the probability that the label predicted by model $h$ is different respect true label.

> In general way, *an error* can allows to compute the fitting of a model by a learning algorithm.

![[Pasted image 20241106130122.png]]


![[Pasted image 20241106130142.png]]

