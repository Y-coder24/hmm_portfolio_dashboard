# 📊 Regime-Based Portfolio Dashboard

Live Demo: [https://hmmportfoliodashboard-zhgvarn4hxnczxi3ks8qy8.streamlit.app/](https://hmmportfoliodashboard-wn9zo2mqzafpekklffbojx.streamlit.app/)

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
- LLM-powered explainability layer (OpenAI API) generating natural-language 
  portfolio commentary and regime retrospectives  
- Interactive Q&A interface for querying backtest results in natural language

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

### 5. LLM-Based Explainability Layer
To make quantitative outputs more interpretable, structured backtest results 
(regime states, Sharpe/drawdown metrics, position weights, historical regime 
segments) are passed to an LLM (OpenAI API) to generate:
- A natural-language summary of current portfolio state and risk profile  
- A retrospective comparison of portfolio performance across the longest 
  historical bull and bear regimes  
- An interactive Q&A feature allowing users to ask follow-up questions about 
  the backtest results  

**Design note:** Only pre-computed structured summaries (not raw price series) 
are passed to the LLM, and prompts explicitly constrain the model to reason 
only from the provided data — reducing hallucination risk and keeping token 
costs low.
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
- AI-generated portfolio commentary and regime retrospective  
- Natural-language Q&A on backtest results

---

## Performance Notes
The Q&A feature is isolated using Streamlit's `@st.fragment` decorator, so 
that submitting a question does not trigger a full re-run of the HMM training 
and MVO backtest pipeline — only the LLM call and its surrounding UI update.

---

## Tech Stack

- Python  
- Streamlit  
- NumPy / Pandas  
- scikit-learn  
- hmmlearn  
- matplotlib  
- yfinance
- OpenAI API

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run hmm_rolling.py

Create a `.streamlit/secrets.toml` file in the project root with:
```toml
OPENAI_API_KEY = "your-openai-api-key"
```
