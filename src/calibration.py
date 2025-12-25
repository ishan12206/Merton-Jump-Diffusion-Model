import numpy as np
from scipy.optimize import minimize
from src.option_pricing import merton_option_price_formula as merton_call_price

def merton_calibration_loss(
    params,
    market_prices,
    strikes,
    maturities,
    S0,
    r
):
    sigma, lam, mu_J, sigma_J = params

    model_prices = np.array([
        merton_call_price(S0, K, T, r, sigma, lam, mu_J, sigma_J)
        for K, T in zip(strikes, maturities)
    ])

    return np.sum((model_prices - market_prices)**2)


def calibrate_merton(
    market_prices,
    strikes,
    maturities,
    S0,
    r,
    initial_guess
):
    bounds = [
        (0.01, 1.0),
        (0.0, 5.0),
        (-1.0, 1.0),
        (0.01, 1.0)
    ]

    result = minimize(
        merton_calibration_loss,
        x0=initial_guess,
        args=(market_prices, strikes, maturities, S0, r),
        bounds=bounds,
        method="L-BFGS-B"
    )

    return result
