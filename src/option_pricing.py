import numpy as np
from src.simulation import simulate_merton_paths
from scipy.stats import norm
import math

def payoff_call(S,K):
    return np.maximum(S - K, 0)
def payoff_put(S,K):
    return np.maximum(K - S, 0)

def price_european_option_mc(paths,S0,K,r,T,option_type = 'call'):
    """
    Price a European option using Monte Carlo simulation of asset paths.
    
    """
    ST = paths[:, -1]
    if option_type == 'call':
        payoff = payoff_call(ST, K)
    elif option_type == 'put':
        payoff = payoff_put(ST, K)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    discounted_payoff = np.exp(-r * T) * payoff
    return np.mean(discounted_payoff)

def merton_option_price_formula(S0,K,r,T,sigma,lam,mu_J,sigma_J,runs = 200,option_type='call'):
    """ 
    Merton's closed-form solution for European option pricing under the jump-diffusion model.
    """
    Kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    price = 0.0
    lambda_p = lam * (1 + Kappa)
    
    for n in range(runs):
        sigma_n = np.sqrt(sigma**2 + (n * sigma_J**2) / T)
        r_n = r - lam * Kappa + (n * np.log(1 + Kappa)) / T
        log_poisson_prob = -lambda_p * T + n * np.log(lambda_p * T) - math.lgamma(n + 1)
        poisson_prob = np.exp(log_poisson_prob)
        
        d1 = (np.log(S0 / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)
        
        if option_type == 'call':
            price += poisson_prob * (S0 * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2))
        elif option_type == 'put':
            price += poisson_prob * (K * np.exp(-r_n * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    return price

