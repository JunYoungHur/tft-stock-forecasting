"""
Markowitz mean-variance portfolio optimization (Modern Portfolio Theory).

Given a table of historical prices (rows = dates, columns = assets), find the
asset weights that maximize the Sharpe ratio. Predicted returns from the model
feed in here as the downstream allocation step.

Usage:
    python src/portfolio.py --prices data/prices.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.01,
                       weight_bounds: tuple = (0.0, 1.0)) -> dict:
    """Maximize Sharpe ratio subject to fully-invested, long-only constraints."""
    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(mean_returns)

    init = np.repeat(1.0 / n, n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple(weight_bounds for _ in range(n))

    def neg_sharpe(w):
        ret = w @ mean_returns
        vol = np.sqrt(w.T @ cov @ w)
        return -(ret - risk_free_rate) / vol

    opt = minimize(neg_sharpe, init, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    w = opt.x
    ret = float(w @ mean_returns)
    vol = float(np.sqrt(w.T @ cov @ w))
    return {
        "weights": w,
        "return": ret,
        "risk": vol,
        "sharpe_ratio": (ret - risk_free_rate) / vol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True, help="CSV: index=date, columns=assets")
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    res = optimize_portfolio(returns)
    print("Optimal weights:", np.round(res["weights"], 4))
    print(f"Expected return: {res['return']:.4f}")
    print(f"Risk (std):      {res['risk']:.4f}")
    print(f"Sharpe ratio:    {res['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
