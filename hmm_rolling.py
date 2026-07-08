# =========================
# Run in terminal:
# streamlit run hmm_rolling.py
# =========================

# This project implements a regime-based portfolio allocation system:
# 1. Detect market regimes (bull vs bear) using HMM on SPY data
# 2. Estimate expected returns / covariance conditional on market regime
# 3. Apply Mean-Variance Optimization (MVO) and/or Risk Parity (RP) for allocation
# 4. Visualize performance, regimes, and portfolio behavior

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from openai import OpenAI
import json

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
- Portfolio optimized conditional on regime (MVO and/or Risk Parity)
- Covariance estimated with Ledoit-Wolf shrinkage, conditional on regime when enough samples exist
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
# Allocation Method Selector
# =========================
alloc_method = st.selectbox(
    "Allocation Method",
    ["Regime MVO", "Regime Risk Parity", "Blend (50/50)"],
    help=(
        "Regime MVO: mean-variance optimization using regime-conditional mean & covariance.\n"
        "Regime Risk Parity: equal risk contribution weights using regime-conditional covariance "
        "(ignores expected returns, so it is less sensitive to noisy mean estimates).\n"
        "Blend: 50/50 average of the two weight vectors."
    )
)

weight_cap = st.slider("Max weight per asset", 0.1, 1.0, 0.4, 0.05)

# =========================
# Start Button (Trigger Execution)
# =========================
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

start = st.button("🚀 Start Analysis")

if start:
    st.session_state.analysis_started = True

if not st.session_state.analysis_started:
    st.info("Please select ETFs, choose an allocation method, and click Start")
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

# =========================
# Weight Solvers
# =========================

def get_weights(mu_vec, cov, w_bounds=(0.0, 1.0)):
    """Analytic MVO-style solution: w ∝ inv(Σ) @ μ, clipped to be long-only."""
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mu_vec

    w = np.maximum(w, 0)

    if np.isnan(w).any() or w.sum() == 0:
        return np.ones_like(w) / len(w)

    return w / w.sum()


def get_risk_parity_weights(cov, w_bounds=(0.0, 0.4)):
    """Equal Risk Contribution (ERC) weights given a covariance matrix.

    Solves for weights such that each asset contributes equally to total
    portfolio variance, subject to long-only + upper bound constraints.
    """
    n = cov.shape[0]

    # Guard against degenerate covariance matrices
    if not np.all(np.isfinite(cov)):
        return np.ones(n) / n

    def risk_contributions(w):
        w = w.reshape(-1, 1)
        port_var = float(w.T @ cov @ w)
        if port_var <= 0:
            return np.zeros(n)
        marginal = cov @ w
        rc = (w * marginal / np.sqrt(port_var)).flatten()
        return rc

    def objective(w):
        rc = risk_contributions(w)
        target = rc.mean()
        return float(np.sum((rc - target) ** 2))

    w0 = np.ones(n) / n
    bounds = [w_bounds] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    try:
        res = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9}
        )
        if res.success and not np.isnan(res.x).any():
            return res.x
    except Exception:
        pass

    return w0  # fallback: equal weight


def get_regime_cov(hist_state, hist_full, cols, min_samples=30):
    """Ledoit-Wolf shrinkage covariance.

    Prefers regime-filtered data if there are enough samples in that regime,
    otherwise falls back to the full rolling window (same behavior as the
    mean estimate's fallback, for consistency).
    """
    if len(hist_state) >= min_samples:
        data = hist_state[cols].values
    else:
        data = hist_full[cols].values

    try:
        lw = LedoitWolf().fit(data)
        cov = lw.covariance_
    except Exception:
        # Fallback to sample covariance if shrinkage fitting fails
        # (e.g. too few rows)
        cov = np.cov(hist_full[cols].values.T)

    return cov


# =========================
# Rolling Allocation (No Look-Ahead)
# =========================
# Use rolling / expanding window to ensure only past data is used

window = 252
cost_rate = 0.001

weights = []
weights_mvo_list = []
weights_rp_list = []
valid_index = []
prev_w = np.ones(len(selected)) / len(selected)

turnover_list = []

for i in range(window, len(returns)):

    hist = returns.iloc[i-window:i]
    state = returns["state"].iloc[i]

    # Regime-filtered history (used for mean AND, when sufficient, covariance)
    hist_state = hist[hist["state"] == state]
    if len(hist_state) < 20:
        hist_state = hist

    # Regime-conditional shrinkage covariance (shared by both methods)
    cov_mat = get_regime_cov(hist_state, hist, selected, min_samples=30)

    # ---- Regime MVO ----
    mu_vec = hist_state[selected].mean().values
    mu_vec = mu_vec * 0.5  # noise dampening
    w_mvo = get_weights(mu_vec, cov_mat)

    # ---- Regime Risk Parity ----
    w_rp = get_risk_parity_weights(cov_mat, w_bounds=(0.0, weight_cap))

    weights_mvo_list.append(w_mvo)
    weights_rp_list.append(w_rp)

    if alloc_method == "Regime MVO":
        w = w_mvo
    elif alloc_method == "Regime Risk Parity":
        w = w_rp
    else:  # Blend
        w = 0.5 * w_mvo + 0.5 * w_rp

    # fallback (critical)
    if np.isnan(w).any() or np.isinf(w).any():
        w = prev_w

    # weight cap
    w = np.clip(w, 0, weight_cap)

    # normalization (safety)
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
weights_mvo_df = pd.DataFrame(weights_mvo_list, index=valid_index, columns=selected)
weights_rp_df = pd.DataFrame(weights_rp_list, index=valid_index, columns=selected)

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

# Trading cost (lagged one day)
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

ax.plot(port_cum, label=f"Portfolio ({alloc_method})")
ax.plot(bh_cum, label="SPY (BH)", linestyle="--")

ax.set_title("Cumulative Return")
ax.legend()

st.pyplot(fig)

# =========================
# Plot 1b: MVO vs Risk Parity vs Blend comparison (informational)
# =========================
st.subheader("⚖️ Method Comparison (informational, not traded simultaneously)")

def quick_backtest(w_df, ret_df, sel, cost_rate_):
    """Cheap comparison backtest: same smoothing/cap already applied upstream
    is NOT reapplied here — this uses the raw per-method weights directly,
    so it's a rough comparison, not identical to the live-traded series."""
    turn = w_df.diff().abs().sum(axis=1).fillna(0)
    g = (w_df * ret_df[sel]).sum(axis=1)
    n = g - turn * cost_rate_
    return np.exp(n.cumsum())

cmp_fig, cmp_ax = plt.subplots(figsize=(12, 5))
cmp_ax.plot(quick_backtest(weights_mvo_df, returns, selected, cost_rate), label="Regime MVO")
cmp_ax.plot(quick_backtest(weights_rp_df, returns, selected, cost_rate), label="Regime Risk Parity")
cmp_ax.plot(bh_cum, label="SPY (BH)", linestyle="--", color="gray")
cmp_ax.set_title("Regime MVO vs Regime Risk Parity vs SPY (unsmoothed weights)")
cmp_ax.legend()
st.pyplot(cmp_fig)

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
st.subheader(f"📊 Portfolio Weights Over Time ({alloc_method})")
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

# =========================
# Identify Regime Segments (for AI recap)
# =========================
def get_regime_segments(state_series, returns_df, port_ret_series):
    """Split the state sequence into contiguous regime segments and
    compute return/drawdown stats for each segment."""
    segments = []
    current_state = state_series.iloc[0]
    start_idx = state_series.index[0]

    for i in range(1, len(state_series)):
        if state_series.iloc[i] != current_state:
            end_idx = state_series.index[i - 1]
            seg_ret = port_ret_series.loc[start_idx:end_idx]
            seg_spy = returns_df["SPY"].loc[start_idx:end_idx]
            segments.append({
                "regime": "Bull" if current_state == 1 else "Bear",
                "start": str(start_idx.date()),
                "end": str(end_idx.date()),
                "days": len(seg_ret),
                "portfolio_return_pct": float(np.expm1(seg_ret.sum()) * 100),
                "spy_return_pct": float(np.expm1(seg_spy.sum()) * 100),
                "portfolio_max_dd_pct": float(max_dd(np.exp(seg_ret.cumsum())) * 100),
            })
            current_state = state_series.iloc[i]
            start_idx = state_series.index[i]

    # Final segment (loop doesn't capture the last one)
    end_idx = state_series.index[-1]
    seg_ret = port_ret_series.loc[start_idx:end_idx]
    seg_spy = returns_df["SPY"].loc[start_idx:end_idx]
    segments.append({
        "regime": "Bull" if current_state == 1 else "Bear",
        "start": str(start_idx.date()),
        "end": str(end_idx.date()),
        "days": len(seg_ret),
        "portfolio_return_pct": float(np.expm1(seg_ret.sum()) * 100),
        "spy_return_pct": float(np.expm1(seg_spy.sum()) * 100),
        "portfolio_max_dd_pct": float(max_dd(np.exp(seg_ret.cumsum())) * 100),
    })
    return segments


all_segments = get_regime_segments(returns["state"], returns, port_ret)

# Keep only the 5 longest bull and 5 longest bear segments to avoid
# an overly long / noisy prompt
bear_segments = sorted(
    [s for s in all_segments if s["regime"] == "Bear"],
    key=lambda x: x["days"], reverse=True
)[:5]
bull_segments = sorted(
    [s for s in all_segments if s["regime"] == "Bull"],
    key=lambda x: x["days"], reverse=True
)[:5]


# =========================
# AI Portfolio Commentary + Regime Retrospective
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a portfolio analytics assistant. You must ONLY use
the structured data provided to you. Never assume external market knowledge,
never speculate about causes not evidenced in the data, and never give
investment advice or recommendations to buy/sell/hold. If data is insufficient
to answer something, say so explicitly."""


@st.cache_data(show_spinner=False)
def generate_full_report(summary_dict, bull_segs, bear_segs):
    prompt = f"""Based on the data below, write two sections:

1. **Current Snapshot** (~120 words): current regime, current allocation
method, how portfolio Sharpe/drawdown compares to SPY buy-and-hold, any
weight concentration risk, one caveat.

2. **Regime Retrospective** (~150 words): compare how the portfolio performed
during the longest bear-market periods vs the longest bull-market periods
listed below. Note any pattern (e.g. did the portfolio protect capital better
in bear regimes than SPY did?).

CURRENT SUMMARY:
{json.dumps(summary_dict, indent=2, default=str)}

LONGEST BEAR REGIME SEGMENTS:
{json.dumps(bear_segs, indent=2)}

LONGEST BULL REGIME SEGMENTS:
{json.dumps(bull_segs, indent=2)}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


summary = {
    "allocation_method": alloc_method,
    "weight_cap": weight_cap,
    "current_regime": "Bull" if returns["state"].iloc[-1] == 1 else "Bear",
    "portfolio_metrics": metrics.set_index("Metric")["Portfolio"].to_dict(),
    "spy_metrics": metrics.set_index("Metric")["SPY"].to_dict(),
    "current_weights": weights.iloc[-1].round(3).to_dict(),
    "avg_turnover": float(turnover_series.mean()),
    "regime_bull_pct": float(returns["state"].mean()),
}

st.subheader("🤖 AI Portfolio Report")
with st.spinner("Generating AI report..."):
    report = generate_full_report(summary, bull_segments, bear_segments)
st.markdown(report)
st.caption(
    "⚠️ AI-generated content is descriptive only, based solely on the data "
    "shown above — not investment advice."
)


# =========================
# Q&A about this portfolio (isolated fragment — won't trigger full rerun)
# =========================
@st.fragment
def qa_section(summary, bear_segments, bull_segments):
    st.subheader("💬 Ask about this portfolio")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    user_question = st.text_input(
        "Ask a question about the current results "
        "(e.g. 'why is the drawdown so large?')"
    )
    ask_btn = st.button("Ask")

    if ask_btn and user_question.strip():
        qa_context = f"""
CURRENT SUMMARY:
{json.dumps(summary, indent=2, default=str)}

BEAR SEGMENTS:
{json.dumps(bear_segments, indent=2)}

BULL SEGMENTS:
{json.dumps(bull_segments, indent=2)}
"""
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=400,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{qa_context}\n\nQuestion: {user_question}"}
                ]
            )
            answer = response.choices[0].message.content

        st.session_state.qa_history.append((user_question, answer))

    for q, a in reversed(st.session_state.qa_history):
        st.markdown(f"**Q: {q}**")
        st.markdown(a)
        st.divider()

qa_section(summary, bear_segments, bull_segments)
