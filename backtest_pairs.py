"""
backtest_pairs.py — Walk-forward pairs trading backtest.

Implements Proposal Sections 5 (Signals & Filter Stack), 6 (Sizing & Costs),
and 7 (Backtest Design) from EquityPairsTradingProposal.md.

── WHAT THIS DOES ──────────────────────────────────────────────────────────

Walk-forward pipeline:
  1. Load SPX-300 daily close + factor ETF data (spx_data.py).
  2. Compute rolling cluster labels (pairs_feature_matrix.py) — Method A/B/fused.
     Labels are re-estimated every `refit_freq` trading days (burn-in = 252d).
     ARI freeze gate prevents churn when clusters are already stable (§3).
  3. At each refit date d0, discover candidate pairs from close[:d0] with
     formation filters: EG ADF p<0.05, 5≤HL≤30d, 63d corr≥0.75, β∈[0.25,4.0]
     (pairs_discovery.py — §4).
  4. Trade pairs within the window (d0, d1] using T+1 alignment:
     β̂_{t−1}, μ̂_{t−1}, σ̂_{t−1} computed from prior-day window; trade at t.
  5. Signal stack (§5):
       Entry:   |Z_t| ≥ 2.0
       Exit:    |Z_t| ≤ 0.5  OR  days_held ≥ 20  OR  |Z_t| > 4.0
  6. Sizing & costs (§6):
       Per-pair notional: target daily σ_PnL = 10 bps → w = 0.001 / σ̂_spread.
       Costs: 3 bps per leg on entry and exit (6 bps round-trip).
  7. Outputs: daily net/gross PnL, equity PNG, metrics JSON, pairs-by-segment CSV.

── WHY FULL HISTORY IN backtest() ─────────────────────────────────────────

Rolling β̂/ẑ need `roll_win` (63d) of lookback BEFORE the trading window opens.
If we slice close to the 63-day trading window, the first ~63 rows produce NaN
z-scores (no prior data), so no trades fire. We therefore pass the full close
history and restrict PnL attribution to [trade_start, trade_end].

── T+1 SIGNAL ALIGNMENT ────────────────────────────────────────────────────

Proposal §7: "use yesterday's signals; execute at today's close."
In rolling_beta_alpha the OLS window ending at t gives β̂_t. We shift the
spread, μ, σ by 1 day so z at t uses stats computed through t−1. Entries
execute at t's close price — no look-ahead.

── USAGE ────────────────────────────────────────────────────────────────────

  # Quarterly refit, 3 pairs/cluster, 2019–2024 OOS
  PYTHONUNBUFFERED=1 python backtest_pairs.py \\
      --method fused --start 2019-01-01 --end 2024-12-31 \\
      --top-k 3 --refit 63 \\
      --out-dir data/bt_fused_q --log-file data/bt_summaries.log

  --method  : a | b | fused  (spectral / beta-space / fused affinity, §3)
  --refit   : 21=monthly, 63=quarterly, 252=yearly
  --top-k   : max pairs kept per cluster per refit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spx_data import load_all as _spx_load_all
from pairs_feature_matrix import rolling_cluster_labels, hungarian_persistence
from pairs_discovery import discover_pairs, Filters


# ── Trading parameters (Proposal §5-6 defaults) ───────────────────────────

@dataclass
class Params:
    # Signal thresholds (§5 primary mean-reversion signal)
    z_entry: float = 2.0       # |Z| ≥ z_entry → open trade
    z_exit: float = 0.5        # |Z| ≤ z_exit  → close trade (reversion)
    time_stop: int = 20        # max holding days before forced exit
    hard_stop: float = 4.0     # |Z| > hard_stop → emergency exit

    # Rolling estimation window (§5)
    roll_win: int = 63         # days for rolling β̂, μ̂, σ̂ (one calendar quarter)

    # Costs (§6 explicit costs: 2–4 bps/leg; default 3)
    cost_bps: float = 3.0      # bps per leg per trade (entry OR exit)
    # Short borrow fee (annualized bps on short leg notional); applied pro-rata daily while position open
    borrow_apr_bps: float = 0.0

    # Sizing (§6 vol targeting: target 10–15 bps/day σ_PnL per pair)
    target_spread_vol_bps: float = 10.0  # bps; w = target / σ̂_spread

    # Capacity and market-impact toggles (Proposal §6)
    # NOTE: These are off by default for continuity. Turn them on to get
    # more realistic capacity/cost drag vs the simplistic constant-bps model.
    portfolio_notional_usd: float = 10_000_000.0  # convert w (fraction) → $ notional
    enable_adv_cap: bool = False                  # cap leg size at ≤ adv_cap_pct of ADV$
    adv_cap_pct: float = 0.01                    # 1% ADV per leg
    enable_impact: bool = False                  # add impact cost ≈ k × sqrt(participation)
    impact_k_bps: float = 15.0                   # slope k for impact model
    # Optional liquidity floor: skip entries when either leg ADV$ < threshold
    min_adv_usd: float = 0.0                      # 0 disables; else e.g., 2e6

    # Earnings blackout (IBES integration)
    # Block entries within N calendar days of either leg's actual earnings date.
    # 0 = disabled (default); 3 = typical calendar buffer around announcement.
    earnings_blackout_days: int = 0
    # When True (default), also force-exit open positions that enter the blackout window.
    # When False, only block NEW entries — existing positions ride through earnings.
    # Entry-only mode avoids churn from forced-exit + immediate re-entry cycles.
    earnings_force_exit: bool = True

    # Kalman filter β estimation (replaces rolling OLS when use_kalman=True)
    # delta: process noise — how fast β can drift per day (higher = faster adaptation)
    #   0.5e-4 = slow drift (stable pairs), 2e-4 = fast drift (volatile pairs)
    # kalman_beta_unc_cap: max allowed β posterior std / |β| ratio before entry is gated
    #   0 = disabled; 0.3 = skip entry when β estimate is >30% uncertain
    use_kalman: bool = False
    kalman_delta: float = 1e-4
    kalman_beta_unc_cap: float = 0.30

    # Remote PC-core replication knobs (S26 series)
    zero_cross_exit: bool = False      # exit when spread crosses mean (z sign flips)
    month_end_forceclose: bool = False # force-close all open positions at segment end
    freeze_beta: bool = False          # hold β fixed at formation-period value
    no_entry_confirm: bool = False     # skip z-turning-back confirmation; pure |z|>=z_entry
    freeze_spread_stats: bool = False  # when freeze_beta: fix mu/sd from formation window
    normalize_prices: bool = False     # anchor both prices to 1.0 at formation start (remote style)
    cooldown_days: int = 0             # min days after exit before re-entry on same pair


# ── Rolling OLS hedge ratio ───────────────────────────────────────────────

def rolling_beta_alpha(pa: pd.Series, pb: pd.Series, win: int) -> pd.DataFrame:
    """Rolling OLS: pa_t = α + β · pb_t over a trailing `win`-day window.

    WHY OLS on log prices?
    Proposal §5: spread_t = log(P1_t) − β̂·log(P2_t) − α̂. OLS on the
    log-price pair gives a hedge ratio β̂ that makes the spread stationary
    (assuming cointegration). The intercept α̂ absorbs the log-price level
    difference so the spread is mean-zero.

    Returns a DataFrame indexed to the END of each rolling window so β̂_t
    represents the estimate from data through day t. Caller MUST shift by 1
    day before using β̂ in trading to satisfy T+1 alignment (§7).
    """
    betas, alphas = [], []
    for i in range(win, len(pa) + 1):
        a_slice = pa.iloc[i - win:i].values
        b_slice = pb.iloc[i - win:i].values
        Xb = np.c_[np.ones(win), b_slice]
        coef = np.linalg.lstsq(Xb, a_slice, rcond=None)[0]
        alphas.append(coef[0])
        betas.append(coef[1])
    idx = pa.index[win - 1:]
    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=idx)


# ── Kalman filter hedge ratio ─────────────────────────────────────────────

def kalman_beta_alpha(
    pa: pd.Series,
    pb: pd.Series,
    delta: float = 1e-4,
    warmup: int = 63,
) -> pd.DataFrame:
    """Kalman filter for dynamic hedge ratio: pa_t = α_t + β_t · pb_t + ε_t.

    State-space model (Elliott et al. 2005):
      State x = [β, α]  — both follow independent random walks
      Transition: x_t = x_{t-1} + w_t,  w_t ~ N(0, Vw)
      Observation: y_t = H_t · x_t + ε_t, ε_t ~ N(0, Ve)

    Key implementation choices vs naive Kalman:
    1. SCALED observation: H = [pb_t/pb_scale, 1] where pb_scale = rolling 30-day
       mean of pb. This keeps |H[0]| ≈ 1 regardless of price level, preventing
       the large log-price magnitudes (≈6-9) from driving Kalman gain too high and
       making β adapt much faster than intended. β is stored in "scaled" units
       internally and converted back via beta_out = beta_internal / pb_scale.
    2. OLS-SEEDED initial P: P_0 = diag(σ²_β_ols, σ²_α_ols) from warmup regression.
       This prevents the over-broad P=I initialization from causing spuriously fast
       early updates that decay beta_std slowly.
    3. Adaptive Ve: rolling 30-day residual variance (re-estimated every step).

    Args:
        pa, pb: log-price series (same index, aligned)
        delta:  process noise per step; controls β drift speed.
                delta=1e-4 ≈ quarterly adaptation (similar to 63d rolling OLS lag).
                delta=1e-6 ≈ annual (slow, for stable pairs).
        warmup: OLS warm-up window; also seeds initial P and Ve.

    Returns DataFrame [beta, alpha, beta_std] indexed same as pa.
      beta, alpha: in ORIGINAL (unscaled) log-price units — drop-in for rolling_beta_alpha.
      beta_std: posterior std of β in original units; for entry uncertainty gating.
    """
    n = len(pa)
    pa_v, pb_v = pa.values, pb.values

    w = min(warmup, max(20, n // 4))

    # Rolling pb scale (30-day mean of pb) — precomputed for the full series.
    # Using trailing mean avoids look-ahead; first `w` values use expanding mean.
    pb_scale_arr = np.empty(n)
    for t in range(n):
        lo = max(0, t - 29)
        pb_scale_arr[t] = np.mean(pb_v[lo:t + 1])
    pb_scale_arr = np.where(np.abs(pb_scale_arr) < 1e-6, 1.0, pb_scale_arr)

    # Scaled pb for the warmup regression
    pb_scaled_w = pb_v[:w] / pb_scale_arr[:w]  # ≈ 1.0 by construction

    # OLS on warmup to seed state and initial covariance
    Xb = np.c_[pb_scaled_w, np.ones(w)]
    b0, res, _, _ = np.linalg.lstsq(Xb, pa_v[:w], rcond=None)  # [β_scaled, α]
    resid_w = pa_v[:w] - Xb @ b0
    Ve_seed = float(np.var(resid_w) + 1e-8)

    # OLS covariance → seed P (proper initial uncertainty from data, not P=I)
    XtX_inv = np.linalg.inv(Xb.T @ Xb + 1e-8 * np.eye(2))
    P = Ve_seed * XtX_inv  # (2×2) state covariance matching OLS precision

    x = b0.copy()          # [β_scaled, α]
    Vw = np.eye(2) * delta

    resid_sq_buf: list = list(resid_w[-30:] ** 2)
    Ve = Ve_seed

    betas, alphas, beta_stds = [], [], []

    for t in range(n):
        sc = pb_scale_arr[t]
        pb_sc = pb_v[t] / sc      # scaled observation regressor ≈ 1.0

        # ── Predict ───────────────────────────────────────────────────────
        P = P + Vw

        # ── Scaled observation vector H = [pb_sc, 1] ──────────────────────
        H = np.array([pb_sc, 1.0])

        resid = pa_v[t] - float(H @ x)
        resid_sq_buf.append(resid ** 2)
        if len(resid_sq_buf) > 30:
            resid_sq_buf.pop(0)
        Ve = float(np.mean(resid_sq_buf)) + 1e-8

        # ── Update ────────────────────────────────────────────────────────
        S = float(H @ P @ H) + Ve
        K = (P @ H) / S
        x = x + K * resid
        P = (np.eye(2) - np.outer(K, H)) @ P

        # Convert from scaled to original units: β_orig = β_scaled / sc
        beta_orig = float(x[0]) / sc
        beta_std_orig = float(np.sqrt(max(P[0, 0], 0.0))) / sc

        betas.append(beta_orig)
        alphas.append(float(x[1]))
        beta_stds.append(beta_std_orig)

    return pd.DataFrame(
        {"beta": betas, "alpha": alphas, "beta_std": beta_stds},
        index=pa.index,
    )


# ── IBES earnings blackout helper ────────────────────────────────────────

def _build_ticker_blackout(
    ibes_df: Optional[pd.DataFrame],
    tickers: list,
    date_index: pd.DatetimeIndex,
    n_days: int,
) -> dict:
    """Return {ticker: frozenset of blocked Timestamps} from IBES earnings dates.

    Blocks a symmetric window of n_days calendar days around each actual
    announcement date. Uses calendar days (not trading days) so weekends
    near earnings are also caught. Only dates present in date_index are kept
    (avoids allocating Timestamps for non-trading days, speeds up lookup).
    """
    blocked: dict = {}
    if ibes_df is None or n_days <= 0:
        return blocked
    ticker_set = set(tickers)
    sub = ibes_df[ibes_df["ticker"].isin(ticker_set)]
    date_set = set(date_index)
    for tkr, grp in sub.groupby("ticker"):
        dates: set = set()
        for ed in grp["earnings_date"].dropna():
            for delta in range(-n_days, n_days + 1):
                t = ed + pd.Timedelta(days=delta)
                if t in date_set:
                    dates.add(t)
        if dates:
            blocked[tkr] = frozenset(dates)
    return blocked


# ── Single-pair backtest ──────────────────────────────────────────────────

def backtest(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    pairs: pd.DataFrame,
    start: str | None,
    end: str | None,
    p: Params,
    ibes_df: Optional[pd.DataFrame] = None,
    regime_scale: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Simulate all pairs in `pairs` over the [start, end] trading window.

    IMPORTANT: `close` must be the FULL history (not pre-sliced to [start,end]).
    Rolling β̂ and z need `roll_win` days of lookback before trading starts.
    PnL is only attributed to dates within [start, end].

    WHY FULL HISTORY?
    Rolling windows (β̂, μ̂, σ̂) need warm-up data before the trading segment.
    If we slice to the 63-day window, the first ~63 rows yield NaN z-scores
    and no trades fire. Passing full history and filtering PnL at the end
    solves this without look-ahead (we cap at `end` to prevent future leakage).

    Returns:
        DataFrame with columns [pnl, cost, pnl_gross] indexed by trading date.
        Empty DataFrame if no pairs produce trades.
    """
    # Cap at trade_end to prevent future data leakage; don't filter start yet
    logp = np.log(close)
    vol_df = volume
    if end:
        logp = logp.loc[:end]
        vol_df = vol_df.loc[:end]

    trade_start_ts = pd.Timestamp(start) if start else logp.index[0]
    trade_end_ts   = pd.Timestamp(end)   if end   else logp.index[-1]

    # Build per-ticker earnings blackout sets from IBES (empty dict if disabled)
    all_tickers = list({r["A"] for _, r in pairs.iterrows()} | {r["B"] for _, r in pairs.iterrows()})
    blackout = _build_ticker_blackout(ibes_df, all_tickers, logp.index, p.earnings_blackout_days)

    pnl_daily  = []
    cost_daily = []
    trade_stats = []

    for _, r in pairs.iterrows():
        a, b = r["A"], r["B"]
        if a not in logp.columns or b not in logp.columns:
            continue
        pa = logp[a].dropna()
        pb = logp[b].dropna()
        idx = pa.index.intersection(pb.index)
        pa, pb = pa.loc[idx], pb.loc[idx]

        # 63d ADV$ for capacity/impact — computed from raw close×volume
        # We keep this simple and local to the pair to avoid heavy precomputation.
        adv_a = (close[a] * volume[a]).rolling(63).mean().reindex(idx) if (a in close.columns and a in volume.columns) else pd.Series(np.nan, index=idx)
        adv_b = (close[b] * volume[b]).rolling(63).mean().reindex(idx) if (b in close.columns and b in volume.columns) else pd.Series(np.nan, index=idx)

        # Need enough history for at least one full rolling window PLUS the
        # trading segment. Without this the z-score is all NaN.
        if len(idx) < p.roll_win + 21:
            continue

        # Hedge ratio estimation
        if p.freeze_beta:
            # Formation-period β frozen for entire trading segment (Gatev 2006 / remote PC-core)
            _row = pairs.loc[(pairs["A"] == a) & (pairs["B"] == b)] if not pairs.empty else pd.DataFrame()
            frozen_b = float(_row["beta"].iloc[0]) if not _row.empty and "beta" in _row.columns else 1.0
            frozen_a = float(_row["alpha"].iloc[0]) if not _row.empty and "alpha" in _row.columns else 0.0
            ba = pd.DataFrame({"beta": frozen_b, "alpha": frozen_a}, index=pa.index)
        elif p.use_kalman:
            ba = kalman_beta_alpha(pa, pb, delta=p.kalman_delta)
        else:
            ba = rolling_beta_alpha(pa, pb, p.roll_win)

        # Optionally anchor both price series to 1.0 at formation start (remote style)
        if p.normalize_prices:
            _norm_date = pa.index[pa.index < trade_start_ts][-1] if any(pa.index < trade_start_ts) else pa.index[0]
            pa = pa / pa.loc[_norm_date]
            pb = pb / pb.loc[_norm_date]

        # Spread: residual of log(P1) − β̂·log(P2) − α̂
        s = pa.loc[ba.index] - (ba["beta"] * pb.loc[ba.index] + ba["alpha"])

        # T+1 alignment (§7): shift β/μ/σ by 1 so stats at t come from t−1.
        ba_lag  = ba.shift(1)
        s_lag   = s.shift(1)
        if p.freeze_spread_stats and p.freeze_beta:
            # Formation-period normalization: mu/sd fixed from data BEFORE trade_start.
            # Matches remote PC-core: spread normalized at construction, z is stable.
            _form = s_lag[s_lag.index < trade_start_ts].dropna()
            _mu = float(_form.mean()) if len(_form) >= 5 else 0.0
            _sd = float(_form.std())  if len(_form) >= 5 else np.nan
            mu = pd.Series(_mu, index=s.index)
            sd = pd.Series(_sd if (_sd and _sd > 0) else np.nan, index=s.index)
        else:
            mu = s_lag.rolling(p.roll_win).mean()
            sd = s_lag.rolling(p.roll_win).std().replace(0, np.nan)
        z = (s - mu) / sd   # today's spread vs μ/σ

        # Pair-return space for sizing and PnL (more coherent units; see note)
        # r_pair_t = r_A_t − β̂_{t−1} · r_B_t; σ_pair = rolling std(r_pair) shifted by 1 day.
        rA = close[a].pct_change().reindex(ba.index)
        rB = close[b].pct_change().reindex(ba.index)
        r_pair = rA - ba_lag["beta"] * rB
        sigma_pair = r_pair.rolling(p.roll_win).std().shift(1).replace(0, np.nan)

        # For Kalman: extract shifted β uncertainty series for entry gating
        beta_std_lag = ba["beta_std"].shift(1) if (p.use_kalman and "beta_std" in ba.columns) else None

        # Restrict trading loop to segment window [trade_start, trade_end]
        trade_idx = ba.index[
            (ba.index >= trade_start_ts) & (ba.index <= trade_end_ts)
        ]
        if trade_idx.empty:
            continue

        # ── Trading loop ────────────────────────────────────────────────
        pos               = 0
        days_in           = 0
        days_since_exit   = p.cooldown_days  # start ready to trade
        entry_s           = None
        entry_beta        = None   # β̂ at entry — for β-drift structural-break exit
        entry_regime_sc   = 1.0    # regime scale locked at entry; 1.0 = no scaling
        prev_t      = None
        prev_z      = np.nan # z_{t-1} — for entry confirmation filter
        recent_rA: list[float] = []  # last 5 days rA for 5d corr check
        recent_rB: list[float] = []
        pair_pnl    = pd.Series(0.0, index=trade_idx)
        pair_cost   = pd.Series(0.0, index=trade_idx)
        round_trips = 0
        wins        = 0
        holding_days: list[int] = []

        for t in trade_idx:
            z_t  = z.loc[t]
            sd_pair_t = sigma_pair.loc[t]
            r_pair_t  = r_pair.loc[t]
            beta_t    = float(ba_lag["beta"].loc[t]) if t in ba_lag.index else np.nan
            rA_t      = float(rA.loc[t]) if t in rA.index and np.isfinite(rA.loc[t]) else 0.0
            rB_t      = float(rB.loc[t]) if t in rB.index and np.isfinite(rB.loc[t]) else 0.0

            # Track 5-day return history for structural-break corr check
            recent_rA.append(rA_t); recent_rB.append(rB_t)
            if len(recent_rA) > 5: recent_rA.pop(0); recent_rB.pop(0)

            if (not np.isfinite(z_t)
                or not np.isfinite(sd_pair_t) or sd_pair_t <= 0
                or not np.isfinite(r_pair_t)):
                prev_t = t; prev_z = z_t
                continue

            # Position size (Proposal §6): vol-targeted fraction of portfolio
            # w0 = (target_bps / 10000) / σ_pair
            w0 = (p.target_spread_vol_bps / 10000.0) / sd_pair_t

            # Apply per-leg ADV cap (≤ adv_cap_pct of ADV$) by scaling w0
            w = w0
            if p.enable_adv_cap and np.isfinite(w0) and w0 > 0:
                notional = p.portfolio_notional_usd
                leg_usd = w0 * notional
                adv_a_t = float(adv_a.loc[t]) if pd.notna(adv_a.loc[t]) else np.nan
                adv_b_t = float(adv_b.loc[t]) if pd.notna(adv_b.loc[t]) else np.nan
                parts = []
                if np.isfinite(adv_a_t) and adv_a_t > 0:
                    parts.append(leg_usd / adv_a_t)
                if np.isfinite(adv_b_t) and adv_b_t > 0:
                    parts.append(leg_usd / adv_b_t)
                if parts:
                    max_part = max(parts)
                    if max_part > 0:
                        scale = min(1.0, p.adv_cap_pct / max_part)
                        w = w0 * scale

            # ── Regime scale (optional HMM overlay) ───────────────────
            # regime_scale is a pd.Series (date → scalar in [0,1]); 1.0 = full
            # size, 0.0 = skip. Applied at entry only; held fixed for trade.
            r_scale = 1.0
            if regime_scale is not None and t in regime_scale.index:
                r_scale = float(regime_scale.loc[t])
                if not np.isfinite(r_scale):
                    r_scale = 1.0

            # ── Entry logic (§5) ──────────────────────────────────────
            # Confirmation filter: require z turning back toward 0 (reduce
            # entries that head straight to hard_stop).
            # Short: z_t < prev_z (spread falling, i.e., reverting from top)
            # Long:  z_t > prev_z (spread rising, i.e., reverting from bottom)
            z_confirming_short = True if p.no_entry_confirm else (np.isfinite(prev_z) and z_t < prev_z)
            z_confirming_long  = True if p.no_entry_confirm else (np.isfinite(prev_z) and z_t > prev_z)

            if pos == 0:
                days_since_exit += 1
                # Regime gate: skip entry when HMM scale is below minimum threshold.
                # This avoids opening zero/near-zero phantom trades that block future entries.
                if r_scale < 0.05:
                    prev_t = t; prev_z = z_t
                    continue

                # Earnings blackout: block new entries near either leg's earnings date
                if blackout and (t in blackout.get(a, frozenset()) or t in blackout.get(b, frozenset())):
                    prev_t = t; prev_z = z_t
                    continue

                # Kalman β uncertainty gate: skip entry when β estimate is unreliable.
                # β_std / |β| > cap → estimation window is unstable (e.g., early Kalman
                # warm-up, or rapid structural regime change). Avoids entering on a
                # stale or uncertain hedge ratio that will mis-size the spread.
                if (beta_std_lag is not None and p.kalman_beta_unc_cap > 0
                        and t in beta_std_lag.index):
                    bs = float(beta_std_lag.loc[t])
                    bt = float(ba_lag["beta"].loc[t]) if t in ba_lag.index else np.nan
                    if np.isfinite(bs) and np.isfinite(bt) and abs(bt) > 1e-6:
                        if bs / abs(bt) > p.kalman_beta_unc_cap:
                            prev_t = t; prev_z = z_t
                            continue

                # Cooldown: min days after last exit before re-entry
                if p.cooldown_days > 0 and days_since_exit < p.cooldown_days:
                    prev_t = t; prev_z = z_t
                    continue

                # Liquidity floor (optional): require both legs ADV$ ≥ min_adv_usd
                if p.min_adv_usd > 0:
                    adv_a_t0 = float(adv_a.loc[t]) if pd.notna(adv_a.loc[t]) else 0.0
                    adv_b_t0 = float(adv_b.loc[t]) if pd.notna(adv_b.loc[t]) else 0.0
                    if (adv_a_t0 < p.min_adv_usd) or (adv_b_t0 < p.min_adv_usd):
                        prev_t = t; prev_z = z_t
                        continue

                if z_t >= p.z_entry and z_confirming_short:
                    # Short spread: sell A (overpriced), buy B×β (underpriced)
                    pos = -1
                    days_in = 0
                    entry_s = s.loc[t]
                    entry_beta = beta_t
                    entry_regime_sc = r_scale  # lock regime scale for this trade
                    ew = w * entry_regime_sc   # effective weight at entry
                    # Entry costs: explicit + market impact proxy
                    explicit = (p.cost_bps / 10000.0) * (ew + ew)  # two legs
                    impact = 0.0
                    if p.enable_impact:
                        notional = p.portfolio_notional_usd
                        adv_a_t = float(adv_a.loc[t]) if pd.notna(adv_a.loc[t]) else np.nan
                        adv_b_t = float(adv_b.loc[t]) if pd.notna(adv_b.loc[t]) else np.nan
                        part_a = (ew * notional / adv_a_t) if (np.isfinite(adv_a_t) and adv_a_t > 0) else 0.0
                        part_b = (ew * notional / adv_b_t) if (np.isfinite(adv_b_t) and adv_b_t > 0) else 0.0
                        impact_bps = p.impact_k_bps * (np.sqrt(part_a) + np.sqrt(part_b))
                        impact = impact_bps / 10000.0
                    c = explicit + impact
                    pair_pnl.loc[t]  -= c
                    pair_cost.loc[t] += c
                elif z_t <= -p.z_entry and z_confirming_long:
                    # Long spread: buy A (underpriced), sell B×β (overpriced)
                    pos = +1
                    days_in = 0
                    entry_s = s.loc[t]
                    entry_beta = beta_t
                    entry_regime_sc = r_scale  # lock regime scale for this trade
                    ew = w * entry_regime_sc   # effective weight at entry
                    explicit = (p.cost_bps / 10000.0) * (ew + ew)
                    impact = 0.0
                    if p.enable_impact:
                        notional = p.portfolio_notional_usd
                        adv_a_t = float(adv_a.loc[t]) if pd.notna(adv_a.loc[t]) else np.nan
                        adv_b_t = float(adv_b.loc[t]) if pd.notna(adv_b.loc[t]) else np.nan
                        part_a = (ew * notional / adv_a_t) if (np.isfinite(adv_a_t) and adv_a_t > 0) else 0.0
                        part_b = (ew * notional / adv_b_t) if (np.isfinite(adv_b_t) and adv_b_t > 0) else 0.0
                        impact_bps = p.impact_k_bps * (np.sqrt(part_a) + np.sqrt(part_b))
                        impact = impact_bps / 10000.0
                    c = explicit + impact
                    pair_pnl.loc[t]  -= c
                    pair_cost.loc[t] += c

            # ── Exit logic (§5 + structural-break exits) ──────────────
            else:
                days_in += 1

                # Structural-break exit 1: 5d rolling corr < 0.5
                # If legs decouple, the spread is no longer mean-reverting.
                struct_break = False
                if len(recent_rA) >= 5:
                    corr_5d = np.corrcoef(recent_rA, recent_rB)[0, 1]
                    if np.isfinite(corr_5d) and corr_5d < 0.5:
                        struct_break = True

                # Structural-break exit 2: β drift > 30% since entry (disabled when freeze_beta)
                if (not p.freeze_beta and entry_beta is not None and np.isfinite(beta_t)
                        and abs(entry_beta) > 1e-6):
                    if abs((beta_t - entry_beta) / entry_beta) > 0.30:
                        struct_break = True

                # Earnings exit: force-close if either leg enters blackout window
                earnings_exit = bool(
                    p.earnings_force_exit
                    and blackout
                    and (t in blackout.get(a, frozenset()) or t in blackout.get(b, frozenset()))
                )

                z_exit_hit = (
                    (p.zero_cross_exit and pos * z_t >= 0)  # spread crossed mean (z flipped sign)
                    or (not p.zero_cross_exit and abs(z_t) <= p.z_exit)
                )
                exit_sig = (
                    z_exit_hit
                    or days_in >= p.time_stop     # time stop
                    or abs(z_t) > p.hard_stop     # hard stop
                    or struct_break               # structural break
                    or earnings_exit              # earnings window
                )

                # Daily P&L: vol-targeted w × regime scale (locked at entry) × position
                wsc = w * entry_regime_sc
                pair_pnl.loc[t] += wsc * pos * r_pair_t

                # Apply daily short-borrow accrual while position is open (if enabled)
                if p.borrow_apr_bps > 0 and pos != 0:
                    daily_borrow = (p.borrow_apr_bps / 10000.0) / 252.0 * abs(wsc)
                    pair_pnl.loc[t] -= daily_borrow
                    pair_cost.loc[t] += daily_borrow

                if exit_sig:
                    if entry_s is not None:
                        rt_pnl = (s.loc[t] - entry_s) * pos * wsc
                        wins += 1 if rt_pnl > 0 else 0
                        holding_days.append(days_in)
                        round_trips += 1
                    pos             = 0
                    days_in         = 0
                    days_since_exit = 0   # reset cooldown
                    entry_regime_sc = 1.0  # reset for next trade
                    # Exit costs: explicit + optional market impact
                    explicit = 2 * (p.cost_bps / 10000.0) * wsc
                    impact = 0.0
                    if p.enable_impact:
                        notional = p.portfolio_notional_usd
                        adv_a_t = float(adv_a.loc[t]) if pd.notna(adv_a.loc[t]) else np.nan
                        adv_b_t = float(adv_b.loc[t]) if pd.notna(adv_b.loc[t]) else np.nan
                        part_a = (wsc * notional / adv_a_t) if (np.isfinite(adv_a_t) and adv_a_t > 0) else 0.0
                        part_b = (wsc * notional / adv_b_t) if (np.isfinite(adv_b_t) and adv_b_t > 0) else 0.0
                        impact_bps = p.impact_k_bps * (np.sqrt(part_a) + np.sqrt(part_b))
                        impact = impact_bps / 10000.0
                    c = explicit + impact
                    pair_pnl.loc[t]  -= c
                    pair_cost.loc[t] += c

            prev_t = t
            prev_z = z_t
        # ── End trading loop ─────────────────────────────────────────

        # Force-close any open position at segment end (remote PC-core behaviour)
        if p.month_end_forceclose and pos != 0 and trade_idx.size > 0:
            t_last = trade_idx[-1]
            wsc = w * entry_regime_sc
            explicit = 2 * (p.cost_bps / 10000.0) * wsc
            pair_pnl.loc[t_last]  -= explicit
            pair_cost.loc[t_last] += explicit
            if entry_s is not None:
                rt_pnl = (s.loc[t_last] - entry_s) * pos * wsc
                wins += 1 if rt_pnl > 0 else 0
                holding_days.append(days_in)
                round_trips += 1
            pos = 0

        # Only record pairs that actually traded (avoids cluttering trade_stats)
        if round_trips > 0 or pair_pnl.abs().sum() > 0:
            pnl_daily.append(pair_pnl)
            cost_daily.append(pair_cost)
            trade_stats.append({
                "pair":          f"{a}|{b}",
                "round_trips":   round_trips,
                "hit_rate":      wins / round_trips if round_trips > 0 else np.nan,
                "avg_hold_days": np.mean(holding_days) if holding_days else np.nan,
            })

    if not pnl_daily:
        return pd.DataFrame()

    pnl  = pd.concat(pnl_daily,  axis=1).sum(axis=1).fillna(0.0)
    cost = pd.concat(cost_daily, axis=1).sum(axis=1).fillna(0.0)
    # pnl is already net (costs deducted in loop); pnl_gross adds them back
    out  = pd.DataFrame({"pnl": pnl, "cost": cost, "pnl_gross": pnl + cost})
    out.attrs["trade_stats"] = trade_stats
    return out


# ── Walk-forward pipeline ─────────────────────────────────────────────────

def backtest_monthly_pipeline(
    method:        str        = "fused",
    start:         str | None = None,
    end:           str | None = None,
    top_k:         int        = 5,
    refit_freq:    int        = 21,
    p:             Params     = None,
    corr_window:   int        = 126,
    beta_window:   int        = 252,
    ridge_alpha:   float      = 10.0,
    ari_threshold: float      = 0.6,
    factor_source: str        = "etf_only",
    neutralize:    bool       = False,
    pca_betas:     Optional[int] = None,
    ortho_factors: Optional[int] = None,
    min_corr:      float      = 0.70,
    min_hl:        float      = 5.0,
    max_hl:        float      = 30.0,
    adf_alpha:     float      = 0.05,
    formation:     int        = 252,
    verbose:       bool       = True,
    use_ibes:      bool       = False,
    ibes_blackout_days: int   = 3,
    # a_coint method parameters
    coint_weight:        float = 0.5,
    coint_prefilter_corr: float = 0.70,
    coint_window:        int   = 252,
    # HMM regime overlay
    regime_scale: Optional[pd.Series] = None,
    # Remote PC-core replication knobs
    no_factor_zscore: bool = False,
    optics_xi: float = 0.05,
    c_use_ols: bool = False,
    use_spx_membership: bool = False,
    b_agglo_threshold: float = 0.4,
) -> dict:
    """Full walk-forward backtest matching Proposal §7 protocol.

    WHY WALK-FORWARD? (§7)
    Static in-sample backtest massively overfits. Walk-forward re-estimates
    clusters and pairs at each refit date using ONLY data available at that
    date, then trades forward until the next refit. This is the correct
    simulation of live operation.

    Segment structure:
      burn_in = 252 trading days (enough for LW correlation + ridge betas)
      refit_dates = [burn_in, burn_in+refit_freq, burn_in+2*refit_freq, ...]
      segment i: discover pairs on close[:d0]; trade on (d0, d1]

    WHY PASS FULL close TO backtest()?
    The trading segment is only `refit_freq` days (e.g. 63). Rolling β̂ needs
    another 63d of warm-up before the first z-score is valid. Slicing close
    to the segment would produce NaN z-scores → no trades. Full close is safe:
    we cap at d1 (no future leakage) and only emit PnL within (d0, d1].
    """
    t0 = time.time()
    if verbose:
        print(f"[bt] Loading SPX data (factor_source={factor_source})...")
    if factor_source == "etf_only":
        close, volume, rets, fclose, frets = _spx_load_all(refresh=False)
    else:
        from data_fetcher import DataFetcher
        df = DataFetcher()
        close, volume, rets, fclose, frets = df.load_all(prefer_factors=factor_source, refresh=False)

    # Load IBES earnings dates for blackout filter
    ibes_df: Optional[pd.DataFrame] = None
    if use_ibes and ibes_blackout_days > 0:
        from data_fetcher import DataFetcher
        _fetcher = DataFetcher()
        _ibes_path = _fetcher.data_dir / "ibes_earnings.parquet"
        if _ibes_path.exists():
            ibes_df = pd.read_parquet(_ibes_path)
            ibes_df["earnings_date"] = pd.to_datetime(ibes_df["earnings_date"])
            if verbose:
                print(f"[bt] IBES loaded: {len(ibes_df)} rows, {ibes_df['ticker'].nunique()} tickers, blackout ±{ibes_blackout_days}d")
        else:
            import warnings as _w
            _w.warn("ibes_earnings.parquet not found; earnings blackout disabled.")

    # Load S&P 500 point-in-time membership for universe filtering
    _spx_membership: Optional[pd.DataFrame] = None
    if use_spx_membership:
        from data_fetcher import DataFetcher as _DF
        _mp = _DF().data_dir / "crsp_spx_membership.parquet"
        if _mp.exists():
            _spx_membership = pd.read_parquet(_mp)
            _spx_membership["start_date"] = pd.to_datetime(_spx_membership["start_date"])
            _spx_membership["end_date"]   = pd.to_datetime(_spx_membership["end_date"])
            if verbose:
                print(f"[bt] S&P 500 membership loaded: {len(_spx_membership)} rows")
        else:
            import warnings as _w
            _w.warn("crsp_spx_membership.parquet not found; membership filter disabled.")

    # Merge blackout days into Params so backtest() picks it up
    if p is None:
        p = Params()
    if ibes_df is not None:
        p = Params(**{**p.__dict__, "earnings_blackout_days": ibes_blackout_days})

    # Keep full history for rolling windows; trade_end caps at `end`
    corr_win = corr_window  # §3 Method A: 126d default; try 189 for more stability
    beta_win = beta_window  # §3 Method B: 252d default; try 504 for more stability
    burn_in  = max(corr_win, beta_win)

    rets_full  = rets.loc[:end]  if end  else rets
    frets_full = frets.loc[:end] if end  else frets

    # Factor preprocessing:
    # 1. Drop FF_RF (risk-free rate, not a risk factor; inflates intercept).
    # 2. Rolling 252d z-score each factor column to normalize across regimes.
    #    Skip when no_factor_zscore=True (remote PC-core uses raw SPY returns).
    if "FF_RF" in frets_full.columns:
        frets_full = frets_full.drop(columns=["FF_RF"])
    if not no_factor_zscore:
        frets_full = (
            frets_full
            .rolling(252, min_periods=63)
            .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False)
            .fillna(0.0)
        )

    if verbose:
        print(
            f"[bt] Computing rolling cluster labels "
            f"(method={method}, corr={corr_win}, beta={beta_win}, refit={refit_freq})..."
        )
    t1 = time.time()

    # rolling_cluster_labels applies ARI freeze gate (§3): if ARI(prev,curr) ≥ 0.6,
    # labels are frozen for the period — no churn from noise in the correlation matrix.
    # Method C control returns: SPY-only for paper-equivalent (market partial-corr),
    # or full frets_full for multi-factor residualisation.
    _c_control = frets_full[["SPY"]] if "SPY" in frets_full.columns else frets_full

    labels_df = rolling_cluster_labels(
        rets_full, frets_full,
        method=method, corr_window=corr_win, beta_window=beta_win,
        refit_freq=refit_freq, w_ret=0.5,
        ari_threshold=ari_threshold, ridge_alpha=ridge_alpha,
        pca_betas=pca_betas, ortho_factors=ortho_factors,
        # a_coint: pass close prices and cointegration params
        close_df=close if method == "a_coint" else None,
        coint_window=coint_window,
        w_coint=coint_weight,
        min_corr_prefilter=coint_prefilter_corr,
        # Method C: partial-correlation clustering
        control_rets=_c_control if method in ("c", "c_optics", "c_dbscan") else None,
        c_optics_xi=optics_xi,
        c_use_ols=c_use_ols,
        b_agglo_threshold=b_agglo_threshold,
    )
    if verbose:
        print(f"[bt] Labels ready in {time.time()-t1:.1f}s")

    # Refit dates: every refit_freq steps after burn-in.
    # These are the dates where new cluster labels are estimated and
    # a new set of pairs is discovered from close[:d0].
    n = len(rets_full)
    refit_indices = list(range(burn_in, n, refit_freq))
    refit_dates   = rets_full.index[refit_indices]

    # Restrict segments to the user-specified trading window
    if start:
        refit_dates = refit_dates[refit_dates >= pd.Timestamp(start)]
    if verbose:
        print(f"[bt] Refit periods: {len(refit_dates)}")

    # Segment i covers [d0, d1): refit at d0, trade until next refit d1
    segments = [
        (refit_dates[i], refit_dates[i + 1] if i + 1 < len(refit_dates) else rets_full.index[-1])
        for i in range(len(refit_dates))
    ]

    all_pnl            = []
    segment_pairs_info = []
    prev_pairs: set    = set()          # pairs that passed formation last segment
    pair_pass_hist: dict = {}           # pair → list of bool (last 3 segments)

    for si, (d0, d1) in enumerate(segments, 1):
        seg_t0 = time.time()
        if verbose:
            print(f"[bt] Segment {si}/{len(segments)}: {d0.date()} → {d1.date()} — discovering pairs...")

        labs = labels_df.loc[d0].dropna()
        if labs.empty:
            continue

        # Filter to active S&P 500 members at formation date d0
        if _spx_membership is not None:
            _active = _spx_membership[
                (_spx_membership["start_date"] <= d0) &
                (_spx_membership["end_date"].isna() | (_spx_membership["end_date"] >= d0))
            ]["ticker"].dropna().unique()
            labs = labs[labs.index.isin(_active)]
            if labs.empty:
                continue

        fret_d0 = frets.loc[:d0] if neutralize else None
        filt = Filters(
            neutralize_factors=neutralize,
            neutralize_ridge_alpha=ridge_alpha,
            min_corr63=min_corr,
            min_hl=min_hl,
            max_hl=max_hl,
            adf_alpha=adf_alpha,
        )
        pairs = discover_pairs(
            close.loc[:d0], labs,
            formation_window=formation,
            top_k_per_cluster=top_k,
            per_cluster_cap=40,
            filt=filt,
            factor_rets=fret_d0,
            prev_pairs=prev_pairs,
        )
        if verbose:
            n_maint = int(pairs["maintenance"].sum()) if not pairs.empty and "maintenance" in pairs.columns else 0
            print(f"[bt]   Pairs found: {len(pairs)} ({n_maint} maintenance)")
        if pairs.empty:
            continue

        # New-pair cap: maintenance pairs fill first; cap new pairs at ≤35% of book.
        # Forces ≥65% maintenance mix once overlap builds, reducing untested-pair noise.
        if "maintenance" in pairs.columns:
            maint_pairs = pairs[pairs["maintenance"]]
            new_pairs   = pairs[~pairs["maintenance"]]
            max_new     = max(int(len(maint_pairs) / 0.65 * 0.35) + 1, top_k)
            pairs = pd.concat([maint_pairs, new_pairs.head(max_new)], ignore_index=True)

        seg_pairs = pairs.copy()
        seg_pairs.insert(0, "segment_start", d0)
        seg_pairs.insert(1, "segment_end",   d1)
        segment_pairs_info.append(seg_pairs)

        # Pass FULL close (not sliced to segment) so rolling windows warm up
        # correctly. backtest() restricts PnL to [d0, d1] internally.
        # NOTE: We pass BOTH close and volume so ADV/impact can be computed.
        pnl_seg = backtest(close, volume, pairs, start=str(d0.date()), end=str(d1.date()), p=p, ibes_df=ibes_df, regime_scale=regime_scale)

        if verbose:
            print(f"[bt]   Segment PnL rows: {len(pnl_seg)} in {time.time()-seg_t0:.1f}s")

        if not pnl_seg.empty:
            stats = pd.DataFrame(getattr(pnl_seg, 'attrs', {}).get('trade_stats', []))
            if not stats.empty:
                seg_pairs = seg_pairs.merge(stats, on='pair', how='left')
                segment_pairs_info[-1] = seg_pairs
            all_pnl.append(pnl_seg)

        # Update pair pass history and prev_pairs for next segment's maintenance gate.
        # Pairs that passed formation this segment are candidates for maintenance next time.
        curr_pair_set = set(pairs["pair"]) if not pairs.empty else set()
        for pair_k in curr_pair_set | set(pair_pass_hist.keys()):
            hist = pair_pass_hist.setdefault(pair_k, [])
            hist.append(pair_k in curr_pair_set)
            # Keep only last 3 segments
            pair_pass_hist[pair_k] = hist[-3:]
        # prev_pairs: pairs with ≥ maint_passes_required passes in last 3 segments
        req = filt.maint_passes_required
        prev_pairs = {p for p, h in pair_pass_hist.items() if sum(h) >= req}

    # ── Aggregate results ─────────────────────────────────────────────
    if not all_pnl:
        if verbose:
            print("[bt] No trades generated across all segments.")
        return {"pnl": pd.DataFrame()}

    pnl = pd.concat(all_pnl).sort_index()

    # Ensure required columns exist (some segments may be empty DataFrames)
    for col in ("pnl", "cost", "pnl_gross"):
        if col not in pnl.columns:
            pnl[col] = 0.0

    daily  = pnl["pnl"]
    ann_mu = daily.mean() * 252
    ann_sd = daily.std()  * np.sqrt(252)
    sharpe = ann_mu / ann_sd if ann_sd > 0 else np.nan

    # Maximum drawdown on cumulative net PnL (§7 metrics)
    eq       = daily.cumsum()
    max_dd   = (eq - eq.cummax()).min()

    seg_pairs_df  = pd.concat(segment_pairs_info) if segment_pairs_info else pd.DataFrame()
    unique_pairs  = set(seg_pairs_df["pair"].unique()) if not seg_pairs_df.empty else set()

    # Pair overlap rate across consecutive segments: average forward retention
    overlap_rate = np.nan
    if not seg_pairs_df.empty:
        by_seg = seg_pairs_df.groupby('segment_start')['pair'].apply(set).sort_index()
        overlaps = []
        keys = list(by_seg.index)
        for i in range(len(keys)-1):
            a_set = by_seg.iloc[i]
            b_set = by_seg.iloc[i+1]
            if len(b_set) > 0:
                overlaps.append(len(a_set & b_set) / len(b_set))
        if overlaps:
            overlap_rate = float(np.mean(overlaps))

    # Hungarian-aligned cluster persistence across consecutive refit dates.
    # Separates true membership churn from K-Means label permutation artifacts.
    # Target: ≥ 0.5 (50% of stocks stay in aligned cluster). Below 0.4 = clustering unstable.
    cluster_persist_scores = []
    for i in range(len(refit_dates) - 1):
        d0, d1 = refit_dates[i], refit_dates[i + 1]
        if d0 in labels_df.index and d1 in labels_df.index:
            prev_l = labels_df.loc[d0].dropna()
            curr_l = labels_df.loc[d1].dropna()
            score = hungarian_persistence(prev_l, curr_l)
            if not np.isnan(score):
                cluster_persist_scores.append(score)
    cluster_persistence = float(np.mean(cluster_persist_scores)) if cluster_persist_scores else np.nan

    # Gross Sharpe (pre-cost)
    gross         = pnl["pnl_gross"] if "pnl_gross" in pnl.columns else pnl["pnl"]
    ann_mu_gross  = gross.mean() * 252
    ann_sd_gross  = gross.std()  * np.sqrt(252)
    sharpe_gross  = ann_mu_gross / ann_sd_gross if ann_sd_gross > 0 else np.nan

    # Aggregate trade stats across all segments
    all_stats     = [ts for seg in all_pnl for ts in seg.attrs.get("trade_stats", [])]
    total_trips   = sum(ts["round_trips"] for ts in all_stats)
    avg_hold      = float(np.nanmean([ts["avg_hold_days"] for ts in all_stats])) if all_stats else np.nan
    avg_hit       = float(np.nanmean([ts["hit_rate"]      for ts in all_stats])) if all_stats else np.nan

    if verbose:
        print(f"[bt] Total elapsed {time.time()-t0:.1f}s")
        ov = f"{overlap_rate:.0%}" if not np.isnan(overlap_rate) else "n/a"
        print(f"[bt] Trips: {total_trips}  AvgHold: {avg_hold:.1f}d  HitRate: {avg_hit:.1%}  Overlap: {ov}")

    return {
        "pnl":               pnl,
        "ann_mu":            ann_mu,
        "ann_sd":            ann_sd,
        "sharpe":            sharpe,
        "ann_mu_gross":      ann_mu_gross,
        "sharpe_gross":      sharpe_gross,
        "max_drawdown":      float(max_dd),
        "segments":          len(segments),
        "segment_pairs":     seg_pairs_df,
        "unique_pairs_count": len(unique_pairs),
        "overlap_rate":        overlap_rate,
        "cluster_persistence": cluster_persistence,  # Hungarian-aligned; target ≥ 0.5
        "total_round_trips":   total_trips,
        "avg_hold_days":       avg_hold,
        "avg_hit_rate":        avg_hit,
    }


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Walk-forward pairs trading backtest (Proposal §7)")
    ap.add_argument("--method",   choices=["a", "b", "fused", "a_coint", "c", "c_optics", "c_dbscan", "b_agglo"], default="fused",
                    help="Clustering method: a=spectral, b=beta-space, fused=blended, a_coint=corr+cointegration affinity")
    ap.add_argument("--start",    type=str, default=None,  help="Trading window start (YYYY-MM-DD)")
    ap.add_argument("--end",      type=str, default=None,  help="Trading window end   (YYYY-MM-DD)")
    ap.add_argument("--top-k",    type=int, default=5,     help="Max pairs per cluster per refit (§4)")
    ap.add_argument("--refit",    type=int, default=21,
                    help="Days between cluster refits: 21=monthly, 63=quarterly, 252=yearly (§7)")
    ap.add_argument("--out-dir",  type=str, default="",    help="Directory for pnl.csv, equity.png, metrics.json")
    ap.add_argument("--log-file", type=str, default="",    help="Append one-line JSON summary here")
    # Iteration flags: z thresholds and cost knobs (speed up tuning)
    ap.add_argument("--z-entry",     type=float, default=None, help="Override z-entry threshold (default 2.0)")
    ap.add_argument("--z-exit",      type=float, default=None,  help="Override z-exit threshold (default 0.5; try 0.25 to harvest earlier)")
    ap.add_argument("--hard-stop",   type=float, default=None,  help="Override hard-stop |z| threshold (default 4.0; try 3.0 to cut tail losses)")
    ap.add_argument("--time-stop",   type=int,   default=None,  help="Override max holding days (default 20; try 15)")
    ap.add_argument("--borrow-apr-bps", type=float, default=0.0, help="Short borrow fee APR in bps applied daily to the short leg (default 0: disabled)")
    ap.add_argument("--cost-bps",    type=float, default=None, help="Override explicit per-leg cost bps (default 3.0)")
    # Market impact and capacity controls
    ap.add_argument("--enable-impact",   action="store_true", help="Enable square-root impact cost model on entry/exit")
    ap.add_argument("--impact-k-bps",    type=float, default=None, help="Square-root impact slope k in bps (default 15 bps)")
    ap.add_argument("--enable-adv-cap",  action="store_true", help="Cap leg size at ≤ adv_cap_pct of ADV$ (scales down weights)")
    ap.add_argument("--adv-cap-pct",     type=float, default=None, help="Max participation per leg as fraction of ADV$ (default 0.01 = 1%%)")
    ap.add_argument("--portfolio-notional", type=float, default=None, help="Portfolio notional USD used to translate weights into $ for ADV/impact")
    ap.add_argument("--target-vol-bps",  type=float, default=None, help="Target daily spread vol in bps for per-pair sizing (default 10 bps)")
    ap.add_argument("--min-adv-usd",     type=float, default=None, help="Liquidity floor: require each leg ADV$ ≥ this to allow entries (default 0 = disabled)")
    ap.add_argument("--zero-cost",   action="store_true",      help="Shortcut: set cost_bps=0 and disable impact")
    ap.add_argument("--corr-window",    type=int,   default=126,       help="Trailing days for LW correlation (Method A, default 126; try 189)")
    ap.add_argument("--beta-window",    type=int,   default=252,       help="Ridge regression window for Method B (default 252; try 504)")
    ap.add_argument("--ridge-alpha",    type=float, default=10.0,      help="Ridge regularisation strength for Method B (default 10; try 25-50)")
    ap.add_argument("--pca-betas",      type=int,   default=None,      help="PCA components on betas before Method B affinity (default None; try 7-10)")
    ap.add_argument("--ortho-factors",  type=int,   default=None,      help="PCA on factor RETURNS before ridge (correct order; try 6-10 for Method B stability)")
    ap.add_argument("--ari-thresh",     type=float, default=0.6,       help="ARI freeze threshold — raise to 0.7/0.75 to hold labels longer")
    ap.add_argument("--factor-source",  type=str,   default="etf_only",
                    choices=["etf_only", "ff_fred_first", "both"],
                    help="Factor set for Method B: etf_only (17 ETFs), ff_fred_first (FF5+MOM+FRED), both (FF+FRED+ETF)")
    ap.add_argument("--neutralize",     action="store_true",
                    help="Avellaneda-Lee mode: residualize returns vs factor set before ADF/HL (requires --factor-source)")
    ap.add_argument("--min-corr",       type=float, default=0.70, help="Min 63d corr for new pairs (default 0.70; try 0.65 with neutralize)")
    ap.add_argument("--min-hl",         type=float, default=5.0,  help="Min half-life days for new pairs (default 5; try 8 to remove fast pairs)")
    ap.add_argument("--max-hl",         type=float, default=30.0, help="Max half-life days for new pairs (default 30; try 20 to remove slow pairs)")
    ap.add_argument("--adf-alpha",      type=float, default=0.05, help="ADF p-value threshold for new pairs (default 0.05)")
    ap.add_argument("--formation",      type=int,   default=252,  help="Formation window in trading days (default 252; try 378=1.5yr)")
    ap.add_argument("--coint-weight",          type=float, default=0.5,
                    help="a_coint: weight of A_coint in fused affinity (0=pure corr, 1=pure coint; default 0.5)")
    ap.add_argument("--coint-prefilter-corr",  type=float, default=0.70,
                    help="a_coint: min LW correlation to run ADF test (default 0.70; reduces O(N²) to ~2k-4k pairs)")
    ap.add_argument("--coint-window",          type=int,   default=252,
                    help="a_coint: trading days of log-price history for ADF test (default 252)")
    ap.add_argument("--use-kalman",      action="store_true",
                    help="Use Kalman filter for dynamic β estimation (replaces rolling OLS)")
    ap.add_argument("--kalman-delta",    type=float, default=1e-4,
                    help="Kalman process noise: β drift speed per day (default 1e-4; try 0.5e-4–5e-4)")
    ap.add_argument("--kalman-unc-cap",  type=float, default=0.30,
                    help="Kalman β uncertainty gate: skip entry if beta_std/|beta| > cap (0=off, 0.30=default)")
    ap.add_argument("--earnings-blackout", type=int, default=0,
                    help="Block entries within N calendar days of IBES earnings date per leg "
                         "(0=disabled; 3=typical; requires ibes_earnings.parquet in data/)")
    ap.add_argument("--no-earnings-exit", action="store_true",
                    help="Entry-block-only mode: skip forced exit of open positions hitting the blackout window")
    ap.add_argument("--regime-overlay", type=str, default="none",
                    choices=["none", "hmm_vix"],
                    help="HMM vol-regime entry gate: none=off, hmm_vix=3-state on realized-vol+HY+T10Y2Y")
    ap.add_argument("--regime-w-calm",     type=float, default=1.0,  help="Position scale in calm regime (default 1.0)")
    ap.add_argument("--regime-w-stressed", type=float, default=0.5,  help="Position scale in stressed regime (default 0.5)")
    ap.add_argument("--regime-w-crisis",   type=float, default=0.0,  help="Position scale in crisis regime (default 0.0)")
    ap.add_argument("--quiet",       action="store_true",      help="Suppress progress output")
    # Remote PC-core replication flags (S26 series)
    ap.add_argument("--zero-cross-exit",    action="store_true", help="Exit when spread sign flips (zero-crossing vs z_exit threshold)")
    ap.add_argument("--month-end-forceclose", action="store_true", help="Force-close all open positions at end of each refit segment")
    ap.add_argument("--freeze-beta",        action="store_true", help="Freeze β at formation-period value (Gatev 2006 style)")
    ap.add_argument("--no-factor-zscore",   action="store_true", help="Skip rolling 252d z-score on factor returns (use raw returns)")
    ap.add_argument("--roll-win",           type=int, default=None, help="Rolling window for β/μ/σ estimation in days (default 63)")
    ap.add_argument("--optics-xi",          type=float, default=None, help="OPTICS xi parameter for cluster extraction (default 0.05; remote uses 0.04)")
    ap.add_argument("--c-use-ols",          action="store_true", help="OLS residualization for Method C (vs Ridge α=10)")
    ap.add_argument("--no-entry-confirm",    action="store_true", help="Skip z-turning-back confirmation filter; enter on pure |z|>=z_entry")
    ap.add_argument("--freeze-spread-stats", action="store_true", help="With --freeze-beta: fix mu/sd from formation window (not rolling)")
    ap.add_argument("--spx-membership",      action="store_true", help="Filter universe to active S&P 500 members at each refit date (~500 stocks)")
    ap.add_argument("--b-agglo-threshold",   type=float, default=0.4, help="Agglomerative distance threshold for b_agglo (remote uses 0.4)")
    ap.add_argument("--normalize-prices",    action="store_true", help="Anchor both price series to 1.0 at formation start (remote style)")
    ap.add_argument("--cooldown-days",       type=int, default=0, help="Min days after exit before re-entry on same pair (remote uses 5)")
    args = ap.parse_args()

    # Load HMM regime scale if requested
    _regime_scale: Optional[pd.Series] = None
    if args.regime_overlay == "hmm_vix":
        from regime_hmm import build_regime_scale
        _hmm_path = Path(__file__).parent / "data" / "hmm_regimes.parquet"
        if not _hmm_path.exists():
            print(f"ERROR: {_hmm_path} not found. Run: uv run python regime_hmm.py")
            return
        _regimes = pd.read_parquet(_hmm_path)
        _regime_scale = build_regime_scale(
            _regimes,
            w_calm=args.regime_w_calm,
            w_stressed=args.regime_w_stressed,
            w_crisis=args.regime_w_crisis,
        )
        if not args.quiet:
            print(f"[regime] HMM overlay loaded: {len(_regime_scale)} days, "
                  f"mean_scale={_regime_scale.mean():.3f}")

    # Build Params instance from CLI overrides
    p = Params(
        z_entry   = args.z_entry   if args.z_entry   is not None else Params.z_entry,
        z_exit    = args.z_exit    if args.z_exit    is not None else Params.z_exit,
        hard_stop = args.hard_stop if args.hard_stop is not None else Params.hard_stop,
        time_stop = args.time_stop if args.time_stop is not None else Params.time_stop,
        borrow_apr_bps = args.borrow_apr_bps,
        cost_bps  = 0.0            if args.zero_cost else (args.cost_bps if args.cost_bps is not None else Params.cost_bps),
        enable_impact    = False   if args.zero_cost else (args.enable_impact or Params.enable_impact),
        impact_k_bps     = Params.impact_k_bps if args.impact_k_bps is None else args.impact_k_bps,
        enable_adv_cap   = args.enable_adv_cap or Params.enable_adv_cap,
        adv_cap_pct      = Params.adv_cap_pct if args.adv_cap_pct is None else args.adv_cap_pct,
        portfolio_notional_usd = Params.portfolio_notional_usd if args.portfolio_notional is None else args.portfolio_notional,
        target_spread_vol_bps  = Params.target_spread_vol_bps if args.target_vol_bps is None else args.target_vol_bps,
        min_adv_usd      = Params.min_adv_usd if args.min_adv_usd is None else args.min_adv_usd,
        earnings_force_exit = not args.no_earnings_exit,
        use_kalman          = args.use_kalman,
        kalman_delta        = args.kalman_delta,
        kalman_beta_unc_cap = args.kalman_unc_cap,
        zero_cross_exit      = args.zero_cross_exit,
        month_end_forceclose = args.month_end_forceclose,
        freeze_beta          = args.freeze_beta,
        roll_win             = args.roll_win if args.roll_win is not None else Params.roll_win,
        no_entry_confirm     = args.no_entry_confirm,
        freeze_spread_stats  = args.freeze_spread_stats,
        normalize_prices     = args.normalize_prices,
        cooldown_days        = args.cooldown_days,
    )

    res = backtest_monthly_pipeline(
        method=args.method, start=args.start, end=args.end,
        top_k=args.top_k,  refit_freq=args.refit, p=p,
        corr_window=args.corr_window, beta_window=args.beta_window,
        ridge_alpha=args.ridge_alpha, ari_threshold=args.ari_thresh,
        factor_source=args.factor_source, neutralize=args.neutralize,
        pca_betas=args.pca_betas, ortho_factors=args.ortho_factors,
        min_corr=args.min_corr, min_hl=args.min_hl, max_hl=args.max_hl, adf_alpha=args.adf_alpha,
        formation=args.formation,
        verbose=not args.quiet,
        use_ibes=args.earnings_blackout > 0,
        ibes_blackout_days=args.earnings_blackout,
        coint_weight=args.coint_weight,
        coint_prefilter_corr=args.coint_prefilter_corr,
        coint_window=args.coint_window,
        regime_scale=_regime_scale,
        no_factor_zscore=args.no_factor_zscore,
        optics_xi=args.optics_xi if args.optics_xi is not None else 0.05,
        c_use_ols=args.c_use_ols,
        use_spx_membership=args.spx_membership,
        b_agglo_threshold=args.b_agglo_threshold,
    )
    pnl = res["pnl"]

    if pnl.empty:
        print("No PnL produced — insufficient pairs or data. Check formation filters in pairs_discovery.py.")
        return

    # ── Console summary ───────────────────────────────────────────────
    print(
        f"Net  — Ann mu: {res['ann_mu']:.4f}  Ann sd: {res['ann_sd']:.4f}  "
        f"Sharpe: {res['sharpe']:.2f}  MaxDD: {res['max_drawdown']:.4f}"
    )
    print(
        f"Gross — Ann mu: {res.get('ann_mu_gross', float('nan')):.4f}  "
        f"Sharpe: {res.get('sharpe_gross', float('nan')):.2f}"
    )
    print(f"Date range: {pnl.index[0].date()} → {pnl.index[-1].date()}")
    cp = res.get('cluster_persistence', float('nan'))
    print(
        f"Pairs: {res.get('unique_pairs_count','?')} unique  "
        f"Trips: {res.get('total_round_trips','?')}  "
        f"AvgHold: {res.get('avg_hold_days', float('nan')):.1f}d  "
        f"HitRate: {res.get('avg_hit_rate', float('nan')):.1%}  "
        f"Overlap: {res.get('overlap_rate', float('nan')):.0%}  "
        f"ClusterPersist(Hungarian): {cp:.0%}" if not np.isnan(cp) else
        f"Pairs: {res.get('unique_pairs_count','?')} unique  Trips: {res.get('total_round_trips','?')}  Overlap: {res.get('overlap_rate', float('nan')):.0%}"
    )

    # ── Persist outputs (§7 deliverables) ────────────────────────────
    out_dir = args.out_dir
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        pnl.to_csv(d / "pnl.csv")

        seg_pairs = res.get("segment_pairs")
        if isinstance(seg_pairs, pd.DataFrame) and not seg_pairs.empty:
            seg_pairs.to_csv(d / "pairs_by_segment.csv", index=False)

        # Equity curve: cumulative net vs gross PnL (§7 deliverable chart)
        eq       = pnl["pnl"].cumsum()
        eq_gross = pnl["pnl_gross"].cumsum() if "pnl_gross" in pnl.columns else None
        plt.figure(figsize=(10, 4))
        plt.plot(eq.index, eq.values, label="net", linewidth=1.5)
        if eq_gross is not None:
            plt.plot(eq_gross.index, eq_gross.values, label="gross", alpha=0.7, linewidth=1)
        plt.axhline(0, color="gray", linewidth=0.5)
        plt.title(f"Equity curve — method={args.method}, refit={args.refit}d, top-k={args.top_k}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL (notional units)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(d / "equity.png", dpi=150)
        plt.close()

        def _f(v): return float(v) if (v is not None and v == v) else None  # nan-safe float

        metrics = {
            "method":             args.method,
            "start":              str(pnl.index[0].date()),
            "end":                str(pnl.index[-1].date()),
            "top_k":              args.top_k,
            "refit":              args.refit,
            "z_entry":            p.z_entry,
            "z_exit":             p.z_exit,
            "cost_bps":           p.cost_bps,
            "zero_cost":          args.zero_cost,
            # Performance
            "ann_mu_net":         _f(res["ann_mu"]),
            "ann_sd":             _f(res["ann_sd"]),
            "sharpe_net":         _f(res["sharpe"]),
            "ann_mu_gross":       _f(res.get("ann_mu_gross")),
            "sharpe_gross":       _f(res.get("sharpe_gross")),
            "max_drawdown":       _f(res["max_drawdown"]),
            # Breadth / turnover diagnostics
            "segments":           int(res.get("segments", 0)),
            "unique_pairs_count": int(res.get("unique_pairs_count", 0)),
            "overlap_rate":        _f(res.get("overlap_rate")),
            "cluster_persistence": _f(res.get("cluster_persistence")),  # Hungarian-aligned; target ≥ 0.5
            "total_round_trips":   int(res.get("total_round_trips", 0)),
            "avg_hold_days":       _f(res.get("avg_hold_days")),
            "avg_hit_rate":        _f(res.get("avg_hit_rate")),
            # Clustering knobs (for experiment log)
            "corr_window":         args.corr_window,
            "beta_window":         args.beta_window,
            "ridge_alpha":         args.ridge_alpha,
            "ari_thresh":          args.ari_thresh,
            "factor_source":       args.factor_source,
            "neutralize":          args.neutralize,
            "earnings_blackout_days": args.earnings_blackout,
            "earnings_force_exit":    p.earnings_force_exit,
            "use_kalman":             p.use_kalman,
            "kalman_delta":           p.kalman_delta if p.use_kalman else None,
            "coint_weight":           args.coint_weight if args.method == "a_coint" else None,
            # HMM regime overlay
            "regime_overlay":         args.regime_overlay,
            "regime_w_calm":          args.regime_w_calm     if args.regime_overlay != "none" else None,
            "regime_w_stressed":      args.regime_w_stressed if args.regime_overlay != "none" else None,
            "regime_w_crisis":        args.regime_w_crisis   if args.regime_overlay != "none" else None,
            "regime_mean_scale":      float(_regime_scale.mean()) if _regime_scale is not None else None,
        }
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved outputs to {d}/")

        # Append a concise row to the research log (EXPERIMENTS.md)
        try:
            exp_path = Path(__file__).resolve().parent / "data" / "EXPERIMENTS.md"
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(exp_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"Date: {pd.Timestamp.today().date()}\n"
                    f"Run ID: {Path(out_dir).name}\n"
                    f"Method: {args.method}\n"
                    f"Params: k={args.top_k}, z={p.z_entry}, cost={p.cost_bps}bps, refit={args.refit}, corr_win={args.corr_window}, HL=[{getattr(args,'min_hl', 'NA')},{getattr(args,'max_hl','NA')}], corr63≥{getattr(args,'min_corr','NA')}\n"
                    f"Results: Gross {metrics.get('sharpe_gross')}, Net {metrics.get('sharpe_net')}, MaxDD {metrics.get('max_drawdown')}, Trips {metrics.get('total_round_trips')}, Hit {metrics.get('avg_hit_rate')}\n"
                    f"Note: \n\n"
                    "────────────────────────────────────────\n\n"
                )
        except Exception as e:
            print(f"[warn] Could not append EXPERIMENTS.md: {e}")

    if args.log_file:
        rec = {
            "method":       args.method,
            "start":        str(pnl.index[0].date()),
            "end":          str(pnl.index[-1].date()),
            "top_k":        args.top_k,
            "refit":        args.refit,
            "ann_mu":       float(res["ann_mu"]),
            "ann_sd":       float(res["ann_sd"]),
            "sharpe":       float(res["sharpe"]),
            "max_drawdown": float(res["max_drawdown"]),
        }
        with open(args.log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"Appended summary to {args.log_file}")


if __name__ == "__main__":
    main()
