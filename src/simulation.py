import numpy as np
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt

def simulate_merton_paths(S0, r, sigma, lam, mu_J, sigma_J, T, n_steps, n_paths, seed = None):
    """
    Simulate asset price paths using the Merton jump-diffusion model.

    Parameters:
    S0 : float
        Initial asset price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the continuous part of the returns.
    lam : float
        Jump intensity (average number of jumps per unit time).
    mu_J : float
        Mean of the jump size (in log terms).
    sigma_J : float
        Standard deviation of the jump size (in log terms).
    T : float
        Time horizon for the simulation.
    n_steps : int
        Number of time steps in the simulation.
    n_paths : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    np.ndarray
        Simulated asset price paths of shape (n_paths, n_steps + 1).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps

    Kappa = np.exp(mu_J + 0.5*sigma_J**2) - 1

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = norm.rvs(size = n_paths)
        N = poisson.rvs(lam * dt, size = n_paths)

        diffusion = (r - lam * Kappa - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

        jump  = np.zeros(n_paths)

        for i in range(n_paths):
            if N[i] > 0:
                J = norm.rvs(loc = mu_J, scale = sigma_J, size = N[i])
                jump[i] = np.sum(J)
        
        paths[:,t] = paths[:,t-1] * np.exp(diffusion + jump)

    return paths


def plot_simulated_paths(paths, T):
    """
    Plot simulated asset price paths.

    Parameters:
    paths : np.ndarray
        Simulated asset price paths of shape (n_paths, n_steps + 1).
    T : float
        Time horizon for the simulation.
    """
    n_steps = paths.shape[1] - 1
    time_grid = np.linspace(0, T, n_steps + 1)

    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(time_grid, paths[i], lw=0.5)
    
    plt.title('Simulated Asset Price Paths using Merton Jump-Diffusion Model')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.grid()
    plt.show()

