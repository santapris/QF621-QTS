"""
pairs_feature_matrix.py — Clustering feature matrix for equity pairs discovery.

Implements EquityPairsTradingProposal.md Section 3 (Clustering for peer discovery):

Method A — returns-spectral:
  1. 126d return correlation matrix C
  2. Ledoit-Wolf shrinkage → Ĉ
  3. k from Marchenko-Pastur upper bound: λ+ = (1+√q)², q=N/T
  4. Graph Laplacian L = D - Ĉ; k smallest eigenvectors (L2-normalized rows)
  5. K-Means on eigenvector embedding (best of 10 seeds)

Method B — factor-beta:
  1. 17-20 factor ETF returns
  2. EWMA-weighted ridge betas per stock (λ_ewma=0.97, 126-252d window)
  3. Scale betas by factor vol (makes distance scale-invariant)
  4. Cosine affinity A_beta = exp(-d_cos²/σ²)
  5. Spectral clustering on A_beta (same k as Method A ± 20%)

Auxiliary feature matrix (Appendix A):
  mom_3m, mom_12m, vol_21d, beta_spy_63d, corr_to_sector_63d, ADV, Amihud

Fusion (optional):
  A_fused = w * A_ret + (1-w) * A_beta; spectral cluster on A_fused

ARI stability gate:
  When ARI(prev_labels, curr_labels) > 0.6 → freeze labels (no churn).

Usage:
    from spx_data import load_all
    from pairs_feature_matrix import build_feature_matrix, cluster_method_a, cluster_method_b

    close, volume, rets, fclose, frets = load_all()
    end_date = rets.index[-1]

    labels_a, k = cluster_method_a(rets, end_date=end_date)
    labels_b = cluster_method_b(rets, frets, end_date=end_date, k=k)
    labels_fused = cluster_fused(rets, frets, end_date=end_date)
    aux = build_aux_features(close, volume, rets, frets, end_date=end_date)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize


# ── Constants ──────────────────────────────────────────────────────────

CORR_WINDOW = 126
BETA_WINDOW = 252
EWMA_LAM = 0.97
RIDGE_ALPHA = 10.0
KMEANS_SEEDS = 10
ARI_FREEZE_THRESHOLD = 0.6
K_MIN, K_MAX = 2, 25
AFFINITY_SIGMA_PCTILE = 50  # median pairwise cosine distance as σ


# ── Shared utilities ───────────────────────────────────────────────────

def _slice_window(
    df: pd.DataFrame,
    end_date: Optional[pd.Timestamp],
    window: int,
    min_obs: Optional[int] = None,
) -> pd.DataFrame:
    """Return trailing `window` rows ending at end_date; drop sparse columns."""
    if end_date is None:
        end_pos = len(df)
    else:
        end_pos = df.index.get_indexer([end_date], method="pad")[0] + 1
    start_pos = max(0, end_pos - window)
    w = df.iloc[start_pos:end_pos]
    if min_obs is None:
        min_obs = window // 2
    valid = w.columns[w.notna().sum() >= min_obs]
    return w[valid].fillna(0.0)


def _lw_correlation(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk correlation matrix from (T × N) returns array."""
    lw = LedoitWolf().fit(R)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-10, 1.0, std)
    C = cov / np.outer(std, std)
    np.fill_diagonal(C, 1.0)
    return C


def _kmeans_best(X: np.ndarray, k: int, seeds: int = KMEANS_SEEDS) -> np.ndarray:
    """K-Means with multiple random seeds; return best-inertia labels."""
    best_labels, best_inertia = None, np.inf
    for seed in range(seeds):
        km = KMeans(n_clusters=k, n_init=1, max_iter=500, random_state=seed)
        labels = km.fit_predict(X)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_labels = labels
    return best_labels


# ── Marchenko-Pastur k selection ───────────────────────────────────────

def mp_upper_bound(N: int, T: int) -> float:
    """MP noise ceiling for correlation matrix eigenvalues.

    λ+ = (1 + √(N/T))²
    Eigenvalues above this carry signal; below is noise.
    """
    q = N / T
    return (1.0 + np.sqrt(q)) ** 2


def select_k_mp(C: np.ndarray, T: int) -> int:
    """Count eigenvalues of C above the MP upper bound → number of clusters."""
    N = C.shape[0]
    lam_plus = mp_upper_bound(N, T)
    eigvals = np.linalg.eigvalsh(C)  # ascending
    k = int(np.sum(eigvals > lam_plus))
    return int(np.clip(k, K_MIN, K_MAX))


# ── Method A: Returns-spectral clustering ─────────────────────────────

def _spectral_embedding(C: np.ndarray, k: int) -> np.ndarray:
    """Compute k-dimensional spectral embedding from correlation matrix C.

    Steps:
      1. Shift C to non-negative affinity A = (C + 1) / 2
      2. Build unnormalized Laplacian L = D - A
      3. Find k smallest eigenvectors of L
      4. L2-normalize rows (standard spectral clustering step)

    Returns (N × k) embedding matrix.
    """
    A = (C + 1.0) / 2.0  # map [-1,1] → [0,1]; keeps relative structure
    d = A.sum(axis=1)
    L = np.diag(d) - A

    # k smallest eigenvectors (ascending order from eigh)
    eigvals, eigvecs = np.linalg.eigh(L)
    V = eigvecs[:, :k]  # (N × k)

    # L2-normalize rows so KMeans distances are meaningful
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return V / norms


def cluster_method_a(
    return_df: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    window: int = CORR_WINDOW,
    k_override: Optional[int] = None,
) -> Tuple[pd.Series, int]:
    """Method A: spectral clustering on LW-shrunk return correlation.

    Args:
        return_df:  (dates × tickers) simple daily returns
        end_date:   formation date; defaults to last available
        window:     trailing days for correlation estimation (126d default)
        k_override: fix k instead of deriving from MP bound

    Returns:
        labels:  pd.Series (ticker → cluster_int)
        k:       number of clusters used
    """
    w = _slice_window(return_df, end_date, window)
    tickers = w.columns.tolist()
    R = w.values  # T × N
    T, N = R.shape

    C = _lw_correlation(R)

    k = k_override if k_override is not None else select_k_mp(C, T)

    V = _spectral_embedding(C, k)
    labels = _kmeans_best(V, k)

    return pd.Series(labels, index=tickers, name="cluster_a"), k


# ── Method B: Factor-beta clustering ──────────────────────────────────

def _ewma_weights(T: int, lam: float = EWMA_LAM) -> np.ndarray:
    """EWMA weights: w_t = λ^(T-1-t), normalized to sum=1, then scaled to T."""
    t = np.arange(T)
    w = lam ** (T - 1 - t)
    w /= w.sum()
    return w * T  # scale so WLS residuals are on original scale


def _orthogonalize_factors(
    factor_rets: pd.DataFrame,
    n_components: int = 8,
) -> pd.DataFrame:
    """PCA on factor returns → orthogonal risk factors.

    WHY ORTHOGONALIZE FIRST (not after betas):
    Our factor set (FF5+MOM + 17 ETFs) is highly collinear: SPY ≈ FF_MktRF,
    XLK ≈ tech loading, TLT ≈ FF_HML, etc. Ridge regression on correlated
    inputs produces unstable betas even with regularization — the alpha
    merely shrinks the instability, it doesn't remove the collinearity.

    Correct order (Barra-style):
      correlated factors → PCA → orthogonal factors → ridge → betas

    This gives betas to independent risk factors (similar to Barra's
    statistical factor model). Betas are much more stable quarter-to-quarter
    because each orthogonal factor represents a distinct risk dimension with
    no overlap.

    Returns:
        DataFrame (dates × n_components) of orthogonal factor returns, columns PC1..PCn
    """
    from sklearn.decomposition import PCA

    X = factor_rets.fillna(0.0).values
    n = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n, random_state=42)
    ortho = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n)]
    return pd.DataFrame(ortho, index=factor_rets.index, columns=cols)


def _ridge_betas_ewma(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    window: int = BETA_WINDOW,
    end_date: Optional[pd.Timestamp] = None,
    ridge_alpha: float = RIDGE_ALPHA,
    ewma_lam: float = EWMA_LAM,
    ortho_factors: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate EWMA-weighted ridge betas for each stock vs factor ETFs.

    When ortho_factors=N: PCA on the factor set first to get N orthogonal
    factors, then regress stocks on those. This resolves collinearity and
    produces more stable betas (correct order: PCA factors → betas, not
    PCA betas → clustering).

    Returns:
        beta_df: (tickers × factors) beta matrix
    """
    sw = _slice_window(stock_rets, end_date, window)
    fw = factor_rets.reindex(sw.index).fillna(0.0)
    fw = fw.dropna(axis=1, how="all").fillna(0.0)

    # Orthogonalize factor returns BEFORE regression.
    # CRITICAL: fit PCA on the full available factor history up to end_date
    # (not just the estimation window), so the orthogonal basis is stable
    # across refits. Using window-only PCA makes PC1 incomparable across quarters.
    if ortho_factors is not None and ortho_factors > 0:
        # Full history up to end_date for PCA basis (T+1 safe: no future data)
        if end_date is not None:
            full_factors = factor_rets.loc[:end_date].fillna(0.0)
        else:
            full_factors = factor_rets.fillna(0.0)
        full_factors = full_factors.dropna(axis=1, how="all").fillna(0.0)
        from sklearn.decomposition import PCA
        n = min(ortho_factors, full_factors.shape[1], full_factors.shape[0] - 1)
        pca = PCA(n_components=n, random_state=42)
        pca.fit(full_factors.values)
        # Transform only the estimation window using the stable basis
        fw_vals = pca.transform(fw[full_factors.columns].fillna(0.0).values)
        fw = pd.DataFrame(fw_vals, index=fw.index,
                          columns=[f"PC{i+1}" for i in range(n)])

    T = len(sw)
    w_sqrt = np.sqrt(_ewma_weights(T, ewma_lam))  # (T,)

    X = fw.values  # T × K
    X_aug = np.c_[np.ones(T), X]  # add intercept column
    X_w = X_aug * w_sqrt[:, np.newaxis]  # WLS transform

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    betas: dict[str, np.ndarray] = {}

    for tkr in sw.columns:
        y = sw[tkr].values
        y_w = y * w_sqrt
        ridge.fit(X_w, y_w)
        betas[tkr] = ridge.coef_[1:]  # drop intercept; (K,)

    beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=fw.columns)
    return beta_df


def _cosine_affinity(beta_df: pd.DataFrame, sigma_pctile: int = AFFINITY_SIGMA_PCTILE) -> np.ndarray:
    """Build affinity matrix A_beta = exp(-d_cos² / σ²) from beta matrix.

    Cosine distance d_cos(i,j) = 1 - cos_similarity(i,j).
    σ² = variance at `sigma_pctile` percentile of pairwise distances.
    """
    B = normalize(beta_df.values, norm="l2")  # L2-normalize rows
    cos_sim = B @ B.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    d_cos = 1.0 - cos_sim  # (N × N) cosine distances in [0, 2]
    np.fill_diagonal(d_cos, 0.0)

    # Adaptive σ: median (or chosen percentile) of upper-triangle distances
    upper = d_cos[np.triu_indices_from(d_cos, k=1)]
    sigma2 = np.percentile(upper, sigma_pctile) ** 2
    if sigma2 < 1e-12:
        sigma2 = 1e-12

    A = np.exp(-d_cos**2 / sigma2)
    np.fill_diagonal(A, 1.0)
    return A


def _spectral_on_affinity(A: np.ndarray, k: int) -> np.ndarray:
    """Spectral clustering on a precomputed affinity matrix A (N × N).

    Build Laplacian from A, take k smallest eigenvectors, K-Means.
    Identical procedure to Method A but with A_beta as input.
    """
    d = A.sum(axis=1)
    L = np.diag(d) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    V = eigvecs[:, :k]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    V = V / norms
    return _kmeans_best(V, k)


def cluster_method_b(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    window: int = BETA_WINDOW,
    k: int = 10,
    k_guard_frac: float = 0.20,
    ridge_alpha: float = RIDGE_ALPHA,
    ewma_lam: float = EWMA_LAM,
    pca_components: Optional[int] = None,
    ortho_factors: Optional[int] = None,
) -> pd.Series:
    """Method B: spectral clustering in EWMA-ridge beta-space.

    k is clipped within ±k_guard_frac of the provided k (from Method A)
    to keep the two views consistent.

    Returns:
        labels: pd.Series (ticker → cluster_int)
    """
    beta_df = _ridge_betas_ewma(
        stock_rets, factor_rets, window=window, end_date=end_date,
        ridge_alpha=ridge_alpha, ewma_lam=ewma_lam, ortho_factors=ortho_factors,
    )

    # Scale betas by factor vol (scale-invariant cosine distance).
    # When ortho_factors is set, PCs are already variance-normalized by PCA
    # so we skip vol-scaling (common_cols would be empty since columns are "PC1"... not ETF names).
    if ortho_factors is not None and ortho_factors > 0:
        beta_scaled = beta_df.fillna(0.0)  # PCs already orthonormal
    else:
        fw = factor_rets.reindex(stock_rets.index).fillna(0.0)
        factor_vol = fw.std().replace(0, np.nan).fillna(1.0)
        common_cols = beta_df.columns.intersection(factor_vol.index)
        beta_scaled = beta_df[common_cols] * factor_vol[common_cols].values

    # Optional PCA on betas: reduces redundant factor blocks (e.g., SPY vs MktRF)
    # and decorrelates the feature space before cosine affinity.
    if pca_components is not None and pca_components < beta_scaled.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components, random_state=42)
        beta_arr = pca.fit_transform(beta_scaled.fillna(0.0).values)
        beta_scaled = pd.DataFrame(beta_arr, index=beta_scaled.index)

    # Guardrail: k within ±guard of Method A k
    k_lo = max(K_MIN, int(k * (1 - k_guard_frac)))
    k_hi = min(K_MAX, int(k * (1 + k_guard_frac)) + 1)
    k_b = int(np.clip(k, k_lo, k_hi))

    A_beta = _cosine_affinity(beta_scaled)
    labels = _spectral_on_affinity(A_beta, k_b)

    return pd.Series(labels, index=beta_scaled.index, name="cluster_b")


# ── Fusion ─────────────────────────────────────────────────────────────

def cluster_fused(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    corr_window: int = CORR_WINDOW,
    beta_window: int = BETA_WINDOW,
    w_ret: float = 0.5,
    k_override: Optional[int] = None,
) -> Tuple[pd.Series, int]:
    """Fuse Method A and B affinities; spectral cluster on blended graph.

    A_fused = w_ret * A_ret + (1 - w_ret) * A_beta

    Returns:
        labels: pd.Series (ticker → cluster_int)
        k:      clusters used
    """
    # --- Method A affinity ---
    w = _slice_window(stock_rets, end_date, corr_window)
    tickers_a = w.columns.tolist()
    R = w.values
    T, N = R.shape
    C = _lw_correlation(R)
    k = k_override if k_override is not None else select_k_mp(C, T)
    A_ret = (C + 1.0) / 2.0  # (N × N)

    # --- Method B affinity ---
    beta_df = _ridge_betas_ewma(stock_rets, factor_rets, window=beta_window, end_date=end_date)
    fw = factor_rets.reindex(stock_rets.index).fillna(0.0)
    factor_vol = fw.std().replace(0, np.nan).fillna(1.0)
    common_cols = beta_df.columns.intersection(factor_vol.index)
    beta_scaled = beta_df[common_cols] * factor_vol[common_cols].values

    # Align to common tickers (Method A may have dropped some)
    common_tickers = [t for t in tickers_a if t in beta_scaled.index]
    if not common_tickers:
        raise ValueError("No common tickers between Method A and B windows.")

    A_ret_s = A_ret[
        [tickers_a.index(t) for t in common_tickers], :
    ][:, [tickers_a.index(t) for t in common_tickers]]
    A_beta_s = _cosine_affinity(beta_scaled.loc[common_tickers])

    # Blend
    A_fused = w_ret * A_ret_s + (1.0 - w_ret) * A_beta_s

    # Clip k to guardrail
    k_b = int(np.clip(k, max(K_MIN, int(k * 0.8)), min(K_MAX, int(k * 1.2) + 1)))
    labels = _spectral_on_affinity(A_fused, k_b)

    return pd.Series(labels, index=common_tickers, name="cluster_fused"), k_b


# ── ARI stability gate ────────────────────────────────────────────────

def apply_ari_freeze(
    prev_labels: Optional[pd.Series],
    curr_labels: pd.Series,
    threshold: float = ARI_FREEZE_THRESHOLD,
) -> pd.Series:
    """Return curr_labels if ARI < threshold (changed enough); else prev_labels.

    On the first call (prev_labels=None), always accept curr_labels.
    Aligns on common tickers; tickers missing from prev get curr value.
    """
    if prev_labels is None:
        return curr_labels

    common = curr_labels.index.intersection(prev_labels.index)
    if len(common) < 2:
        return curr_labels

    ari = adjusted_rand_score(prev_labels.loc[common].values, curr_labels.loc[common].values)

    if ari >= threshold:
        # Cluster stable enough — freeze (propagate prev labels)
        merged = prev_labels.copy()
        new_tickers = curr_labels.index.difference(prev_labels.index)
        merged = pd.concat([merged, curr_labels.loc[new_tickers]])
        return merged
    else:
        return curr_labels


# ── Auxiliary feature matrix ──────────────────────────────────────────

def build_aux_features(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    sector_etf_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build Appendix A auxiliary feature matrix for each stock.

    Features:
        mom_3m              63d cumulative return
        mom_12m             252d cumulative return
        vol_21d             21d annualized return volatility
        beta_spy_63d        OLS beta to SPY over 63d
        corr_to_sector_63d  63d correlation to assigned sector ETF
        adv_63d             63d average dollar volume (log-scaled)
        amihud_21d          21d Amihud illiquidity ratio (×10^6 for readability)

    Args:
        close_df:       (dates × tickers) adjusted close
        volume_df:      (dates × tickers) volume
        stock_rets:     (dates × tickers) simple daily returns
        factor_rets:    (dates × ETFs) factor ETF daily returns
        end_date:       formation date (default: last row)
        sector_etf_map: {ticker: sector_etf} e.g. {'AAPL': 'XLK'}; optional

    Returns:
        aux_df: (tickers × 7) feature DataFrame
    """
    if end_date is None:
        end_pos = len(stock_rets)
    else:
        end_pos = stock_rets.index.get_indexer([end_date], method="pad")[0] + 1

    def _tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[max(0, end_pos - n):end_pos]

    rets_63 = _tail(stock_rets, 63)
    rets_252 = _tail(stock_rets, 252)
    rets_21 = _tail(stock_rets, 21)
    close_63 = _tail(close_df, 63)
    vol_63 = _tail(volume_df, 63)

    tickers = stock_rets.columns.tolist()
    features: dict[str, pd.Series] = {}

    # mom_3m: 63d compound return
    features["mom_3m"] = (1 + rets_63).prod() - 1

    # mom_12m: 252d compound return (may be shorter if near start)
    features["mom_12m"] = (1 + rets_252).prod() - 1

    # vol_21d: annualized volatility
    features["vol_21d"] = rets_21.std() * np.sqrt(252)

    # beta_spy_63d: simple OLS beta to SPY
    spy_63 = factor_rets["SPY"].reindex(rets_63.index).fillna(0.0).values
    spy_var = np.var(spy_63)
    betas_spy = {}
    for tkr in tickers:
        y = rets_63[tkr].fillna(0.0).values
        if spy_var > 1e-12:
            betas_spy[tkr] = float(np.cov(y, spy_63)[0, 1] / spy_var)
        else:
            betas_spy[tkr] = np.nan
    features["beta_spy_63d"] = pd.Series(betas_spy)

    # corr_to_sector_63d
    if sector_etf_map is not None:
        corr_sector: dict[str, float] = {}
        for tkr in tickers:
            etf = sector_etf_map.get(tkr)
            if etf and etf in factor_rets.columns:
                etf_63 = factor_rets[etf].reindex(rets_63.index).fillna(0.0)
                tkr_63 = rets_63[tkr].fillna(0.0)
                c = np.corrcoef(tkr_63.values, etf_63.values)
                corr_sector[tkr] = float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0
            else:
                corr_sector[tkr] = np.nan
        features["corr_to_sector_63d"] = pd.Series(corr_sector)
    else:
        features["corr_to_sector_63d"] = pd.Series(np.nan, index=tickers)

    # adv_63d: average daily dollar volume (log)
    dollar_vol_63 = close_63 * vol_63
    adv = dollar_vol_63.mean()
    features["adv_63d"] = np.log1p(adv)

    # amihud_21d: mean(|return| / dollar_volume) × 1e6
    dv_21 = _tail(close_df, 21) * _tail(volume_df, 21)
    r_21 = rets_21.abs()
    dv_21 = dv_21.reindex(r_21.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amihud = (r_21 / dv_21.replace(0, np.nan)).mean() * 1e6
    features["amihud_21d"] = amihud

    aux_df = pd.DataFrame(features).loc[tickers]
    return aux_df


# ── Walk-forward clustering with ARI freeze ───────────────────────────

def hungarian_persistence(prev_labels: pd.Series, curr_labels: pd.Series) -> float:
    """Fraction of stocks staying in the same cluster after optimal label alignment.

    K-Means cluster numbers are arbitrary permutations each run. ARI handles
    this but gives a scalar similarity, not an interpretable retention rate.
    Hungarian alignment finds the best label mapping, then counts how many
    stocks retained their cluster membership under that mapping.

    Returns a float in [0, 1]: 1.0 = perfectly stable, 0.0 = complete churn.
    """
    from scipy.optimize import linear_sum_assignment

    common = prev_labels.index.intersection(curr_labels.index)
    if len(common) < 2:
        return np.nan

    pv = prev_labels.loc[common].values.astype(int)
    cv = curr_labels.loc[common].values.astype(int)
    pk, ck = np.unique(pv), np.unique(cv)
    pm = {v: i for i, v in enumerate(pk)}
    cm = {v: i for i, v in enumerate(ck)}

    # Contingency matrix: C[i,j] = stocks in prev-cluster i AND curr-cluster j
    C = np.zeros((len(pk), len(ck)), dtype=int)
    for p_, c_ in zip(pv, cv):
        C[pm[p_], cm[c_]] += 1

    row_ind, col_ind = linear_sum_assignment(-C)  # maximise overlap
    return float(C[row_ind, col_ind].sum()) / len(common)


def rolling_cluster_labels(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    method: str = "fused",
    corr_window: int = CORR_WINDOW,
    beta_window: int = BETA_WINDOW,
    refit_freq: int = 21,
    w_ret: float = 0.5,
    ari_threshold: float = ARI_FREEZE_THRESHOLD,
    ridge_alpha: float = RIDGE_ALPHA,
    pca_betas: Optional[int] = None,
    ortho_factors: Optional[int] = None,
) -> pd.DataFrame:
    """Compute cluster labels for every refit date; apply ARI freeze gate.

    Args:
        method:      'a' | 'b' | 'fused'
        refit_freq:  trading days between re-estimations (default 21 ≈ monthly)

    Returns:
        label_df: (dates × tickers) cluster label ints; NaN before first valid window
    """
    burn_in = max(corr_window, beta_window)
    n_dates = len(stock_rets)
    tickers = stock_rets.columns.tolist()

    label_df = pd.DataFrame(
        np.nan, index=stock_rets.index, columns=tickers, dtype="float64"
    )
    prev_labels: Optional[pd.Series] = None
    current_labels: Optional[pd.Series] = None

    refit_indices = range(burn_in, n_dates, refit_freq)

    for idx in refit_indices:
        end_date = stock_rets.index[idx]
        try:
            if method == "a":
                new_labels, k = cluster_method_a(stock_rets, end_date=end_date, window=corr_window)
            elif method == "b":
                _, k = cluster_method_a(stock_rets, end_date=end_date, window=corr_window)
                new_labels = cluster_method_b(stock_rets, factor_rets, end_date=end_date, k=k,
                                              window=beta_window, ridge_alpha=ridge_alpha,
                                              pca_components=pca_betas,
                                              ortho_factors=ortho_factors)
            else:
                new_labels, k = cluster_fused(
                    stock_rets, factor_rets, end_date=end_date,
                    corr_window=corr_window, beta_window=beta_window, w_ret=w_ret,
                )
        except Exception as e:
            warnings.warn(f"Clustering failed at {end_date}: {e}; freezing previous labels.")
            new_labels = current_labels if current_labels is not None else None

        if new_labels is not None:
            current_labels = apply_ari_freeze(prev_labels, new_labels, ari_threshold)
            prev_labels = current_labels

        if current_labels is not None:
            next_idx = min(idx + refit_freq, n_dates)
            for tkr, lab in current_labels.items():
                if tkr in label_df.columns:
                    label_df.loc[label_df.index[idx:next_idx], tkr] = lab

    return label_df


if __name__ == "__main__":
    from spx_data import load_all

    print("Loading data...")
    close, volume, rets, fclose, frets = load_all()

    end_date = rets.index[-1]
    print(f"Formation date: {end_date.date()}, universe: {rets.shape[1]} stocks")

    print("\nMethod A (returns-spectral)...")
    labels_a, k = cluster_method_a(rets, end_date=end_date)
    print(f"  k={k} clusters | {labels_a.value_counts().sort_index().to_dict()}")

    print("\nMethod B (factor-beta)...")
    labels_b = cluster_method_b(rets, frets, end_date=end_date, k=k)
    print(f"  clusters | {labels_b.value_counts().sort_index().to_dict()}")

    ari = adjusted_rand_score(
        labels_a.reindex(labels_b.index).dropna().values,
        labels_b.dropna().values,
    )
    print(f"  ARI(A, B) = {ari:.3f}")

    print("\nFused clustering...")
    labels_f, k_f = cluster_fused(rets, frets, end_date=end_date)
    print(f"  k={k_f} clusters | {labels_f.value_counts().sort_index().to_dict()}")

    print("\nAuxiliary features...")
    aux = build_aux_features(close, volume, rets, frets, end_date=end_date)
    print(aux.describe().round(3))
