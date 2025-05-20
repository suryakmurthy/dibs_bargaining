import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

INVESTMENT_WINDOWS = {
    "5D": 5,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "2Y": 730,
    "5Y": 1825,
    "8Y": 2920
}


def from_subspace_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """
    Forward transform: (n-1)-dimensional vector -> n-dimensional simplex
    """
    last_coord = 1.0 - torch.sum(v)
    return torch.cat((v, last_coord.unsqueeze(0)), dim=0)


def from_simplex_to_subspace(w: torch.Tensor) -> torch.Tensor:
    """
    Inverse transform: n-dimensional simplex -> (n-1)-dimensional vector
    """
    return w[:-1]


def project_gradient_to_simplex_tangent(grad: torch.Tensor) -> torch.Tensor:
    """
    Project a gradient vector onto the tangent space of the simplex,
    i.e., remove the component orthogonal to the simplex.
    """
    return grad - grad.mean()


def from_subspace_to_simplex_batch(v: torch.Tensor) -> torch.Tensor:
    """
    Batched version of forward transform: (n-1)-dimensional -> n-dimensional simplex.

    Args:
        v (torch.Tensor): Tensor of shape [B, d-1]

    Returns:
        torch.Tensor: Tensor of shape [B, d] mapped to simplex
    """
    last_coord = 1.0 - v.sum(dim=1, keepdim=True)  # shape: [B, 1]
    return torch.cat((v, last_coord), dim=1)  # shape: [B, d]


def from_simplex_to_subspace_batch(w: torch.Tensor) -> torch.Tensor:
    """
    Batched version of inverse transform: n-dimensional simplex -> (n-1)-dimensional subspace.

    Args:
        w (torch.Tensor): Tensor of shape [B, d]

    Returns:
        torch.Tensor: Tensor of shape [B, d-1]
    """
    return w[:, :-1]


def project_to_subsimplex(v):
    """
    Project v ∈ R^{n-1} into the subsimplex: v_i >= 0, sum(v) <= 1
    """
    v = torch.clamp(v, min=0.0)
    s = torch.sum(v)
    if s > 1.0:
        v = v / s
        v = v * 1.0
    return v


def compute_subspace_gradient_with_barrier(v, Sigma, lambda_mu, barrier_coeff=1e-6):
    """
    Compute gradient of loss + barrier terms in the subspace.
    """
    v = v.clone().detach().requires_grad_(True)
    w = from_subspace_to_simplex(v)

    quad = torch.dot(w, Sigma @ w)
    linear = torch.dot(lambda_mu, w)
    loss = quad - linear

    eps = 1e-6
    barrier = -torch.sum(torch.log(w + eps)) - torch.log(1.0 - torch.sum(v) + eps)

    total_loss = loss + barrier_coeff * barrier
    total_loss.backward()

    return v.grad


def compute_simplex_gradient(w, Sigma, lambda_mu):
    """
    Compute gradient of just the quadratic loss (no barrier).
    """
    w = w.clone().detach().requires_grad_(True)
    loss = torch.dot(w, Sigma @ w) - torch.dot(lambda_mu, w)
    loss.backward()
    return w.grad


def angle_between(v1, v2):
    """
    Compute angle between two vectors in radians.
    """
    dot_product = torch.dot(v1, v2)
    m1 = torch.norm(v1)
    m2 = torch.norm(v2)
    cos_theta = dot_product / (m1 * m2)
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta_clamped)


def sample_from_simplex(n):
    """
    Sample a point uniformly at random from the n-dimensional simplex.
    """
    return np.random.dirichlet(np.ones(n))


def sample_random_ranges_and_lambdas(n_samples,
                                     min_year=2015,
                                     max_year=2023,
                                     min_range_days=7,
                                     max_range_days=5 * 365,
                                     lambda_range=(0.0, 0.1)):
    """
    Sample random time ranges and lambda values.
    """
    start_date_list, end_date_list, lambda_vals = [], [], []

    for _ in range(n_samples):
        range_days = np.random.randint(min_range_days, max_range_days)
        latest_start = datetime(max_year, 12, 31) - timedelta(days=range_days)
        start_ordinal = datetime(min_year, 1, 1).toordinal()
        end_ordinal = latest_start.toordinal()
        start_date = datetime.fromordinal(np.random.randint(start_ordinal, end_ordinal))
        end_date = start_date + timedelta(days=range_days)

        lam = round(np.random.uniform(*lambda_range), 4)

        start_date_list.append(start_date.strftime("%Y-%m-%d"))
        end_date_list.append(end_date.strftime("%Y-%m-%d"))
        lambda_vals.append(lam)

    return start_date_list, end_date_list, lambda_vals


def sample_investor_windows_and_lambdas(n_samples,
                                        latest_date="2023-12-31",
                                        lambda_range=(0.0, 0.1)):
    """
    Sample date windows from predefined investor horizons ending on latest_date.
    """
    latest = datetime.strptime(latest_date, "%Y-%m-%d")
    window_keys = list(INVESTMENT_WINDOWS.keys())

    start_date_list, end_date_list, lambda_vals = [], [], []

    for _ in range(n_samples):
        # Randomly select one of the investor-style windows
        window_label = np.random.choice(window_keys)
        range_days = INVESTMENT_WINDOWS[window_label]

        start_date = latest - timedelta(days=range_days)
        lam = round(np.random.uniform(*lambda_range), 4)

        start_date_list.append(start_date.strftime("%Y-%m-%d"))
        end_date_list.append(latest.strftime("%Y-%m-%d"))
        lambda_vals.append(lam)

    return start_date_list, end_date_list, lambda_vals


def load_sliced_data(ticker, sampled_start, sampled_end, cache_dir="full_ticker_cache"):
    """
    Load cached ticker data and slice by date.
    """
    filename = f"{ticker}_2015-01-01_2023-12-31.csv"
    filepath = os.path.join(cache_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cached file for {ticker} not found at {filepath}")

    df = pd.read_csv(filepath, skiprows=[1, 2], index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    df.index.name = "Date"
    if "Adj Close" not in df.columns:
        if "Close" not in df.columns:
            raise ValueError(f"No usable price data for {ticker}")
        df["Adj Close"] = df["Close"]

    sliced = df.loc[sampled_start:sampled_end]
    if sliced.empty:
        raise ValueError(f"No data available for {ticker} between {sampled_start} and {sampled_end}")

    return sliced


def setup_markowitz_environment_cached(tickers, start_date, end_date, lambda_val):
    """
    Load cached data and compute returns, covariance matrix, and scaled expected return.
    """
    price_data = []
    for ticker in tickers:
        df = load_sliced_data(ticker, start_date, end_date)
        price_data.append(df["Adj Close"])

    df_all = pd.concat(price_data, axis=1)
    df_all.columns = tickers
    returns = df_all.pct_change().dropna()
    Sigma = returns.cov().values
    mu = returns.mean().values
    lambda_mu = lambda_val * mu
    return Sigma, lambda_mu, tickers