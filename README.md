# multi_loss_regression
Script to study general linear regression problems with different possible losses
Given $n$ data $x_1, \dots, x_n \in \mathbb R^{pk \times k}$, the problem writes:
$$
\sum_{i=1}^n x_i f_i(x_i ^T W) + \| W\|^2, \quad W\in  \mathbb R^{pk},
$$
where $\forall i\in [n]$, $f_i :  \mathbb R^{k} \mapsto \mathbb R^{k}$ is suposedly convex (our prediction works in cases when $f_i$ is non convex but some convergence issues can then appear)


