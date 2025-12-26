# Merton Jump–Diffusion: Tail Risk and Implied Volatility

This project implements and analyzes the **Merton jump–diffusion model** to study how rare but severe price jumps affect return distributions, tail risk measures, and option-implied volatility. The analysis highlights why diffusion-only models such as Black–Scholes underestimate downside risk and fail to explain observed volatility skews in option markets.

---

##  Key Questions Addressed

- How do jumps alter return distributions compared to geometric Brownian motion?
- Why does Black–Scholes severely underestimate tail risk?
- How do jump characteristics map into **VaR**, **CVaR**, and **implied volatility smiles/skews**?
- Which jump parameters control *risk severity* versus *option price distortions*?

---

##  Models

### Black–Scholes (Baseline)
- Continuous diffusion with constant volatility
- Gaussian log-returns
- Zero skewness and kurtosis
- Flat implied volatility surface

### Merton Jump–Diffusion
- Diffusion + Poisson jump component
- Log-normal jump sizes
- Captures discontinuities, crash risk, and heavy tails

The risk-neutral drift is adjusted to preserve the martingale property.

---

##  Analysis Overview

### 1. Path Simulation and Return Properties
- Monte Carlo simulation of price paths
- Comparison of return moments
- Jump–diffusion exhibits:
  - Extreme negative skewness
  - Explosive kurtosis
  - Heavy downside tails

### 2. Tail Risk Analysis (VaR & CVaR)
- Risk measured on **simple returns / losses**
- 99% VaR shows limited sensitivity to rare jumps
- 99% CVaR increases dramatically under jump risk

**Key insight:**  
> *Jump risk primarily affects the severity of extreme losses rather than the loss threshold itself.*

### 3. Option Pricing and Implied Volatility
- European option prices computed under the Merton model
- Prices inverted into Black–Scholes implied volatility
- Results:
  - Pronounced volatility smile
  - Strong left skew for downside strikes
  - Flat volatility assumption breaks down

---

##  Parameter Sensitivity Insights

- **Jump Intensity (λ):**
  - Controls frequency of jumps
  - Increases overall implied volatility level
  - Raises CVaR substantially

- **Jump Mean (μ<sub>J</sub>):**
  - Controls direction and asymmetry of jumps
  - Drives volatility skew
  - More negative μ<sub>J</sub> → steeper left skew

> **Jump frequency controls curvature; jump direction controls skew.**

---

##  Core Takeaways

- Black–Scholes can match variance but fails in the tails
- VaR alone is insufficient to capture jump risk
- CVaR and option prices are highly sensitive to rare extreme events
- Implied volatility skew is the market’s pricing of tail risk

---

