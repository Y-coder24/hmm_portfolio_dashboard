# 📊 Regime-Based Portfolio Dashboard

Live Demo: https://hmmportfoliodashboard-zhgvarn4hxnczxi3ks8qy8.streamlit.app/

---

## Overview

This project implements a regime-based portfolio allocation strategy using a Gaussian Hidden Markov Model (HMM) and mean-variance optimization (MVO).

The model identifies latent market regimes (bull vs bear) using SPY returns and volatility, and dynamically adjusts portfolio weights across multiple ETFs.

---

## Key Features

- Regime detection using Gaussian HMM trained on market-level features  
- Walk-forward training with periodic retraining to avoid look-ahead bias  
- Rolling mean-variance optimization using only historical data  
- State-conditioned expected return estimation  
- Stability enhancements including shrinkage, weight constraints, and smoothing  
- Transaction cost modeling based on portfolio turnover  
- Interactive Streamlit dashboard for real-time visualization  

---

## Methodology

### 1. Market Regime Detection

- Features: SPY log returns and 20-day rolling volatility  
- Model: Gaussian HMM with 2 hidden states  
- Training: Expanding window with retraining every 21 trading days  

---

### 2. Portfolio Optimization

At each time step:

- Estimate expected returns using rolling historical data  
- Condition on current market regime  
- Compute optimal weights using mean-variance optimization:

\[
w \propto \Sigma^{-1} \mu
\]

---

### 3. Stability Enhancements

To address instability in MVO:

- Shrink expected returns  
- Apply long-only constraints and weight caps  
- Smooth portfolio weights over time  
- Implement fallback mechanisms for numerical robustness  

---

### 4. Transaction Cost Modeling

- Turnover is computed as the absolute change in weights  
- Transaction cost is applied as:

\[
\text{cost} = \text{turnover} \times \text{cost rate}
\]

---

## Results

Compared to a SPY buy-and-hold benchmark:

- Improved risk-adjusted performance (higher Sharpe ratio)  
- Lower volatility and drawdowns  
- More stable portfolio behavior  
- Slight reduction in raw return in exchange for improved robustness  

---

## Dashboard Features

- ETF selection interface  
- Cumulative return comparison  
- SPY price with inferred market regimes  
- Time-varying portfolio weights  
- Drawdown visualization  
- Performance metrics (Return, Volatility, Sharpe, Max Drawdown)  

---

## Tech Stack

- Python  
- Streamlit  
- NumPy / Pandas  
- scikit-learn  
- hmmlearn  
- matplotlib  
- yfinance  

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run hmm_rolling.py
