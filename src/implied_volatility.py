import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from src.option_pricing import merton_option_price_formula as merton_call_price

def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    """Calculate the Black-Scholes price of a European option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def implied_vol(
    price,
    S0,
    K,
    r,
    T,
    option_type='call',
    vol_lower=1e-6,
    vol_upper=5.0
):
    """
    Compute Black–Scholes implied volatility for a call option.
    """

    def objective(sigma):
        return black_scholes_price(S0, K, T, r, sigma, option_type=option_type) - price

    return brentq(objective, vol_lower, vol_upper)



