# Ground Truth: Spatial Count Data (spatial_1)

## True generative model

- **Grid:** 12×12 = 144 spatial locations, indexed by (row, col) ∈ {0,…,11}²
- **Likelihood:** Poisson(μ[i])
- **Mean function:** log(μ[i]) = 0.5 + u[i]
- **Spatial random effect:** u[i] = iCAR background (σ=0.2) + two Gaussian hotspots
  - Hotspot 1: centre (row=2, col=2), amplitude=3.0, radius=2.8 grid cells
  - Hotspot 2: centre (row=9, col=9), amplitude=3.0, radius=2.8 grid cells
  - u centred to zero mean

## Oracle NLPD

Oracle NLPD (true μ known, theoretical lower bound): **1.6665**
*(Note: oracle NLPD is stochastic due to the small test set)*

## Baseline NLPDs (fitted Stan, 4 chains × 1000 samples)

| Model | NLPD | Note |
|-------|------|------|
| Oracle (true μ) | 1.49 | Theoretical lower bound |
| Spatial iCAR fitted (Stan) | 1.65 | Correct model class |
| Naive Poisson (no spatial) | 2.35 | Expected agent plateau |

## Gap summary

- Naive Poisson → Spatial iCAR: **+0.70 nats** (clear signal of spatial structure)
- Naive → Oracle: **+0.86 nats** (total discoverable signal)

## What the agent must discover

1. Count data → Poisson likelihood (not Gaussian)
2. Spatial random effects — neighbouring cells have correlated counts
3. Two distinct high-count hotspot regions in the grid
4. iCAR or GP prior using the adjacency structure (node1, node2)

## Key Stan hints (not shown to agent)

```stan
// iCAR log-prior
target += -0.5 * dot_self(u[node1] - u[node2]);
// Sum-to-zero soft constraint
sum(u) ~ normal(0, 0.001 * N_grid);
// Likelihood
y[i] ~ poisson_log(alpha + sigma_u * u[grid_idx[i]]);
```
