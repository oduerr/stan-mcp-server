# Dataset Properties: synthetic_hierarchical_sparse

## Generative Process

```
tau ~ HalfCauchy(0, 2)        # wide hyperprior -> funnel geometry
mu_j ~ Normal(0, tau)         for j = 1..50
y_ij ~ Normal(mu_j, sigma)    for i = 1..3  (train)  / 1..50 (test)
```

**True parameters:**
- `tau = 3.0` (clipped from HalfCauchy draw, seed 20260428)
- `sigma = 0.5`
- `J = 50` groups
- 3 train obs/group, 50 test obs/group

## Key Challenge: Neal's Funnel

With only 3 observations per group, the posterior on `tau` is highly uncertain and
explores values near 0. At small `tau`, the group means `mu_j` must stay near 0
(tight funnel neck), but HMC cannot take appropriately small steps globally.

**Centered parameterization** (`mu_j ~ Normal(0, tau)` directly):
- ~300–600 divergences, elevated R-hat
- Inflated NLPD despite "correct" model structure
- The diagnostic channel is the only signal that something is broken

**Non-centered parameterization (NCP)** (`mu_j = tau * z_j`, `z_j ~ Normal(0,1)`):
- 0 divergences, good R-hat
- Dramatically better NLPD, close to oracle

## Oracle

Oracle NLPD = **0.8617**

Computed analytically: posterior on `mu_j` given 3 training obs and known `tau`, `sigma`
is `Normal(mu_post_j, var_post)` where:
- `var_post = 1 / (1/tau^2 + 3/sigma^2)`
- `mu_post_j = var_post * sum(train_j) / sigma^2`
- Predictive: `Normal(mu_post_j, sigma^2 + var_post)`

## Expected Agent Trajectory

| Model | Divergences | NLPD |
|---|---|---|
| Pooled regression (baseline) | 0 | ~1.8–2.0 |
| Centered hierarchical | 300–600 | ~1.1–1.5 (bad mixing) |
| **NCP hierarchical** | 0 | ~0.87–0.92 |
| Oracle (known τ, posterior μⱼ) | — | 0.8617 |

The diagnostic channel (divergence count + R-hat) is causally necessary here:
the centered model appears to "work" statistically but is broken geometrically.
Only by reading divergences does the agent know to reparameterize.
