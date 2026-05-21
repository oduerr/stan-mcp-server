# Ground Truth: Count Regression

## True generative model

- **Likelihood**: Negative Binomial with overdispersion φ = 3
- **Mean function**: log(μ) = 0.5 + 0.3·x  →  μ = exp(0.5 + 0.3·x)
- **Overdispersion**: φ = 3 (Var = μ + μ²/φ, clearly exceeds Poisson)

## Oracle NLPD

Oracle NLPD (true params): **2.8217**

## Baseline NLPDs (reference)

| Model | NLPD |
|-------|------|
| Gaussian linear | ~3.43 |
| Poisson log-linear | ~3.68 |
| Negative Binomial (oracle) | 2.82 |

## What the agent must discover

1. Response is non-negative integer counts → count likelihood required
2. Variance >> mean (overdispersion) → Negative Binomial, not Poisson
3. Log-linear mean function: `log(mu) = alpha + beta * predictor`
