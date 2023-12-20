# multi_loss_regression
Script to study general linear regression problems with different possible losses
Given $n$ data $x_1, \dots, x_n \in \mathbb R^{pk \times k}$, the problem writes:
$$\underset{W\in  \mathbb R^{pk}}{\text{Minimize}}: \frac 1n\sum_{i=1}^n x_i f_i(x_i ^T W) + \lVert W\rVert^2,$$
where $\forall i\in [n]$, $f_i :  \mathbb R^{k} \mapsto \mathbb R^{k}$ is suposedly convex (our predictions work in cases when $f_i$ is non convex but some convergence issues can then appear)


