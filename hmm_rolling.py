# =========================
# Run in terminal:
# streamlit run hmm_rolling.py
# =========================

# This project implements a regime-based portfolio allocation system:
# 1. Detect market regimes (bull vs bear) using HMM on SPY data
# 2. Estimate expected returns conditional on market regime
# 3. Apply mean-variance optimization (MVO) for asset allocation
# 4. Visualize performance, regimes, and portfolio behavior

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

st.title("📊 Regime-Based Portfolio Dashboard")

# =========================
# Model Description
# =========================
st.markdown("""
### 🧠 Model Design
- HMM trained ONLY on SPY (market regime detection)
- Walk-forward training (expanding window)
- Retrained every 21 trading days
- Cached to avoid recomputation
- Portfolio optimized conditional on regime
""")

# =========================
# ETF Selection
# =========================
all_tickers = [
    "SPY","QQQ","IWM","EFA","EEM",
    "TLT","IEF","LQD","HYG",
    "GLD","SLV","DBC",
    "VNQ","XLE","XLK"
]

selected = st.multiselect(
    "Select ETFs",
    all_tickers,
    default=["SPY","QQQ","TLT","GLD"]
)

# Ensure at least two assets are selected
if len(selected) < 2:
    st.warning("Select at least 2 ETFs")
    st.stop()

# =========================
# Start Button (Trigger Execution)
# =========================
start = st.button("🚀 Start Analysis")

if not start:
    st.info("Please select ETFs and click Start")
    st.stop()

# =========================
# SPY Data (for HMM training)
# =========================
# SPY is used as a proxy for overall market conditions
spy_data = yf.download("SPY", start="2005-01-01")["Close"]

# Compute log returns and rolling volatility
spy_ret = np.log(spy_data / spy_data.shift(1)).dropna()
vol = spy_ret.rolling(20).std()

# Feature matrix: return + volatility
feature = pd.concat([spy_ret, vol], axis=1).dropna()

# Preserve original index for alignment later
feature_raw = feature.copy()
X = feature.values

# =========================
# HMM: Regime Detection
# =========================
# Identify latent market regimes using Gaussian HMM
# Walk-forward training ensures realistic out-of-sample behavior

def run_hmm_with_progress(X):

    initial_train = 1000
    states = []
    model = None

    total_steps = len(X) - initial_train

    # Progress bar for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, t in enumerate(range(initial_train, len(X))):

        train = X[:t]

        # Retrain model every 21 days (monthly frequency)
        if (t - initial_train) % 21 == 0 or model is None:
            model = GaussianHMM(
                n_components=2,
                covariance_type="full",
                n_iter=200,
                random_state=42
            )
            model.fit(train)

        # Predict hidden state
        states_seq = model.predict(X[:t+1])

        # Resolve label switching:
        # Assign "bull" to the state with higher mean return
        means = model.means_[:, 0]
        order = np.argsort(means)
        bull_state = order[-1]

        states.append(1 if states_seq[-1] == bull_state else 0)

        # Update progress bar
        progress = int((i + 1) / total_steps * 100)
        progress_bar.progress(progress)
        status_text.text(f"HMM training: {progress}%")

    return states

# =========================
# Cache HMM Results
# =========================
# Avoid retraining when ETF selection changes
@st.cache_data
def run_hmm_cached(X):
    return run_hmm_with_progress(X)

# =========================
# Control Training Logic
# =========================
if "states" not in st.session_state:
    st.write("### 🔄 Training HMM (first time)...")
    states = run_hmm_with_progress(X)
    st.session_state.states = states
    st.success("✅ HMM training completed!")
else:
    st.info("⚡ Using cached HMM")
    states = st.session_state.states

# =========================
# ETF Data
# =========================
data = yf.download(selected, start="2005-01-01")["Close"]

# Handle single-column case
if isinstance(data, pd.Series):
    data = data.to_frame()

# Remove unavailable tickers
available = data.columns.tolist()
missing = list(set(selected) - set(available))

if missing:
    st.warning(f"Removed missing tickers: {missing}")

selected = available

# Compute log returns
returns = np.log(data / data.shift(1)).dropna()

# =========================
# Data Alignment (Critical Step)
# =========================
# Align ETF data with SPY-based feature index
returns = returns.reindex(feature_raw.index).dropna()

# Align HMM states with return data
states = pd.Series(states, index=feature_raw.index[1000:])

common_index = returns.index.intersection(states.index)
returns = returns.loc[common_index]
states = states.loc[common_index]

returns["state"] = states

def get_weights(mu_vec, cov):
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mu_vec

    w = np.maximum(w, 0)

    if np.isnan(w).any() or w.sum() == 0:
        return np.ones_like(w) / len(w)

    return w / w.sum()


# =========================
# Mean-Variance Optimization (MVO)
# =========================
# =========================
# Mean-Variance Optimization (No Look-Ahead)
# =========================
# Use rolling / expanding window to ensure only past data is used

window = 252
cost_rate = 0.001

weights = []
valid_index = []
prev_w = np.ones(len(selected)) / len(selected)

turnover_list = []

for i in range(window, len(returns)):

    hist = returns.iloc[i-window:i]
    state = returns["state"].iloc[i]

    # 同状态历史
    hist_state = hist[hist["state"] == state]
    if len(hist_state) < 20:
        hist_state = hist

    # μ（降噪）
    mu_vec = hist_state[selected].mean().values
    mu_vec = mu_vec * 0.5

    # Σ
    cov_mat = np.cov(hist[selected].T)

    # MVO
    w = get_weights(mu_vec, cov_mat)

    # fallback（关键）
    if np.isnan(w).any() or np.isinf(w).any():
        w = prev_w

    # weight cap
    w = np.clip(w, 0, 0.4)

    # normalization（安全）
    if w.sum() == 0:
        w = prev_w
    else:
        w = w / w.sum()

    # smoothing
    w = 0.8 * prev_w + 0.2 * w

    # turnover
    turnover = np.sum(np.abs(w - prev_w))
    turnover_list.append(turnover)

    weights.append(w)
    valid_index.append(returns.index[i])

    prev_w = w

# DataFrame
weights = pd.DataFrame(weights, index=valid_index, columns=selected)

# clean
weights = weights.clip(lower=0)
weights = weights.div(weights.sum(axis=1), axis=0)
weights = weights.fillna(1/len(weights.columns))

# align returns
returns = returns.loc[valid_index]

turnover_series = pd.Series(turnover_list, index=valid_index)


# =========================
# Portfolio Performance
# =========================
gross_ret = (weights * returns[selected]).sum(axis=1)

# 交易成本（滞后一日）
cost = turnover_series * cost_rate

net_ret = gross_ret - cost

port_ret = net_ret
bh_ret = returns["SPY"]

port_cum = np.exp(port_ret.cumsum())
bh_cum = np.exp(bh_ret.cumsum())

# =========================
# Plot 1: Cumulative Return
# =========================
fig, ax = plt.subplots(figsize=(12,6))

ax.plot(port_cum, label="Portfolio")
ax.plot(bh_cum, label="SPY (BH)", linestyle="--")

ax.set_title("Cumulative Return")
ax.legend()

st.pyplot(fig)

# =========================
# Plot 2: SPY Price + Regime
# =========================
st.subheader("📈 SPY Price with Market Regimes")

spy_price = spy_data.loc[returns.index]

fig3, ax3 = plt.subplots(figsize=(12,6))

# SPY price
ax3.plot(spy_price, color="black", label="SPY Price")

# Overlay regimes (green = bull, red = bear)
state_series = returns["state"]

for i in range(1, len(state_series)):
    color = "green" if state_series.iloc[i] == 1 else "red"
    ax3.axvspan(
        state_series.index[i-1],
        state_series.index[i],
        color=color,
        alpha=0.1
    )

ax3.set_title("SPY Price with Market Regimes (Green=Bull, Red=Bear)")
ax3.legend()

st.pyplot(fig3)

# =========================
# Portfolio Weights
# =========================
st.subheader("📊 Portfolio Weights")
fig_w, ax_w = plt.subplots(figsize=(12,6))

ax_w.stackplot(
    weights.index,
    weights.T,
    labels=weights.columns
)

ax_w.legend(loc='upper left')
ax_w.set_title("Portfolio Weights Over Time")

st.pyplot(fig_w)

# =========================
# Drawdown
# =========================
dd = port_cum / port_cum.cummax() - 1
st.subheader("📉 Drawdown")
st.line_chart(dd)

# =========================
# Performance Metrics
# =========================
def sharpe(x):
    return np.sqrt(252) * x.mean() / x.std()

def max_dd(cum):
    peak = cum.cummax()
    return (cum/peak - 1).min()

metrics = pd.DataFrame({
    "Metric": ["Return","Vol","Sharpe","Max DD"],
    "Portfolio": [
        port_ret.mean()*252,
        port_ret.std()*np.sqrt(252),
        sharpe(port_ret),
        max_dd(port_cum)
    ],
    "SPY": [
        bh_ret.mean()*252,
        bh_ret.std()*np.sqrt(252),
        sharpe(bh_ret),
        max_dd(bh_cum)
    ]
})

st.subheader("📋 Performance Metrics")
st.dataframe(metrics)
