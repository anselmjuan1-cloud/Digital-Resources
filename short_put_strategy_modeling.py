import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import brentq

def simulate_gbm(
    S0,
    mu,
    sigma,
    T,
    n_steps,
    n_paths=1,
    random_seed=None
    ):

    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps

    # Generate random shocks
    Z = np.random.normal(0, 1, size=(n_steps, n_paths))

    # Preallocate price array
    prices = np.zeros((n_steps + 1, n_paths))
    prices[0] = S0

    # GBM recursion
    for t in range(1, n_steps + 1):
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
        )

    return [p[0] for p in prices]


def black_scholes_put(S, K, T, r, sigma):

    # Calculate d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Black-Scholes formula for put
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price

def put_strike_for_target_delta(
    S,
    sigma,
    r,
    target_delta=-0.20,
    DTE=20
):

    T = DTE / 252  # convert days to years

    def put_delta(K):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1) - 1

    # Reasonable strike bounds
    K_min = 0.01 * S
    K_max = 3.0 * S

    return brentq(
        lambda K: put_delta(K) - target_delta,
        K_min,
        K_max
    )

def simulate_strategy(S0, mu, sigma, implied_sigma, r, 
                      T, n_steps, put_delta, put_dte):
    
    prices = simulate_gbm(S0, mu, sigma, T, n_steps)
    
    net_profit_loss = 0
    trade_outcomes = []
    exp_date = None
    put_strike = None
    for i in range(1, len(prices)):
        price = prices[i]
        if exp_date == None:
            put_strike = put_strike_for_target_delta(price, implied_sigma, 
                                                     r, target_delta=put_delta,
                                                     DTE=put_dte)
            put_price = black_scholes_put(price, put_strike,
                                          put_dte/252, r, implied_sigma)
            net_profit_loss += put_price * 100
            exp_date = i + put_dte
        elif i == exp_date:
            if put_strike < price:
                trade_outcomes.append(1)
                pass
            elif put_strike > price:
                net_profit_loss -= (put_strike - price) * 100
                trade_outcomes.append(0)
            exp_date = None
    
    if exp_date is not None:
        final_price = prices[-1]
        if put_strike > final_price:
            net_profit_loss -= (put_strike - final_price)
            trade_outcomes.append(0)
        else:
            trade_outcomes.append(1)
    
    return net_profit_loss, trade_outcomes

if __name__ == '__main__':
    
    # Define variables
    S0 = 100
    mu = 0.08
    sigma = 0.2
    implied_sigma = sigma * 5
    r = 0.03
    T = 1
    n_steps = 252
    put_delta = -0.05
    put_dte = 20
    
    returns = []
    trades = []
    for i in range(500):
        net_profit_loss, trade_outcomes = (simulate_strategy(S0, mu, sigma,
                                         implied_sigma, r, T,
                                         n_steps, put_delta,
                                         put_dte))
        returns.append(net_profit_loss)
        trades.append(trade_outcomes)
    
    print('Mean Return:', np.mean(returns))
    print('STD Return:', np.std(returns))
    print('Trade-by-Trade Win Rate:', np.mean(trades))
    


