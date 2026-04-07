# Vine decomposition

A vine copula factorizes a high-dimensional dependence model into a sequence of pair copulas and
conditional distributions. In `torchvinecopulib`, the public multivariate entrypoint is
[`VineCop`](../api/vinecop.md), while each pair-copula factor is represented by
[`BiCop`](../api/bicop.md).

For observations $u \in [0, 1]^d$, the joint density can be written as a product of pair-copula
terms:

$$
c(u_1, \dots, u_d) = \prod_{\ell=1}^{d-1} \prod_{e \in T_\ell}
c_{j_e, k_e \mid D_e}\left(
u_{j_e \mid D_e},
u_{k_e \mid D_e}
\right).
$$

The builder path learns:

- the tree sequence $T_\ell$,
- the conditioning sets $D_e$,
- and the pair-copula estimators attached to each edge.

The engine path then executes the resulting plan on tensors. This separation matters because
`fit()` is a structure-learning routine, while `log_pdf()` and `rosenblatt()` are differentiable
query routines.

## Pseudo-observations

Intermediate conditional uniforms are often called pseudo-observations. The query engine stores
them in a flat slot-based memory pool instead of a Python `dict[tuple, Tensor]` structure. This
keeps execution deterministic and avoids dynamic graph assembly during `log_pdf()` and sampling.

## Practical interpretation

- `mtd_vine` controls the tree family (`cvine`, `dvine`, `rvine`).
- `mtd_bidep` controls how candidate edges are weighted during structure learning.
- `bicop_backend` controls how each bivariate copula is estimated and queried.

The factorization is only meaningful when the marginal scale is explicit. If inputs are not
already on copula scale, use `is_cop_scale=False` and let [`VineCop`](../api/vinecop.md) fit or
apply marginal transforms first.
