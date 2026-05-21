"""
data_fetcher.py — Unified data access for the pairs-trading sandbox.

Goals
- Prefer robust sources (Ken French, FRED, WRDS/CRSP/Compustat) when available
- Fall back to yfinance only if caches/primary sources are unavailable
- Persist everything to ./data as parquet for reproducibility

Status
- Implemented: Ken French daily 5 factors + Momentum; FRED macro series; yfinance fallback for
  SPX prices/volume and factor ETFs; simple SPX membership via Wikipedia/yfinance.
- Stubs: WRDS/CRSP/BBG hooks — left as NotImplementedError until credentials are available.

Usage
    from data_fetcher import DataFetcher
    df = DataFetcher()
    stock_close, stock_volume, stock_returns, factor_close, factor_returns = df.load_all(
        prefer_factors="ff_fred_first", refresh=False
    )

Notes
- All network calls are guarded by allow_network; set allow_network=False to use only local caches.
- Aligns factor_returns to stock_returns date index when possible.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataFetcher:
    data_dir: Path = Path(__file__).resolve().parent / "data"
    allow_network: bool = True

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Core caches
        self.spx_close_pq = self.data_dir / "spx_close.parquet"
        self.spx_vol_pq = self.data_dir / "spx_volume.parquet"
        self.factor_close_pq = self.data_dir / "factor_close.parquet"
        self.ff_daily_pq = self.data_dir / "ff_daily.parquet"
        self.fred_pq = self.data_dir / "fred.parquet"

    # ── Ken French daily factors (+ Momentum) ─────────────────────────────────
    def fetch_ff_daily(self, refresh: bool = False) -> pd.DataFrame:
        if self.ff_daily_pq.exists() and not refresh:
            ff = pd.read_parquet(self.ff_daily_pq)
            # Validate cached file has correct columns (not data values)
            if all(c.strip().lstrip("-0123456789.") != "" for c in ff.columns[:3]):
                return ff
            # Bad cache — re-download
        if not self.allow_network:
            raise FileNotFoundError("ff_daily.parquet missing and network disabled")

        import requests

        def _parse_ff_csv(text: str) -> pd.DataFrame:
            """Parse Ken French CSV: skip commentary, detect header+data by date format."""
            lines = text.splitlines()
            # Find first data row (8-digit date in first CSV field)
            data_start = None
            for i, ln in enumerate(lines):
                first = ln.split(",")[0].strip()
                if first.isdigit() and len(first) == 8:
                    data_start = i
                    break
            if data_start is None:
                raise ValueError("No data rows found in FF CSV")
            # Header is the last non-blank, non-digit line before data
            header_idx = data_start - 1
            while header_idx >= 0 and not lines[header_idx].strip():
                header_idx -= 1
            # Collect data rows (stop at blank or non-8-digit date)
            data_lines = []
            for ln in lines[data_start:]:
                first = ln.split(",")[0].strip()
                if not first or not (first.isdigit() and len(first) == 8):
                    break
                data_lines.append(ln)
            csv_text = lines[header_idx] + "\n" + "\n".join(data_lines)
            df = pd.read_csv(io.StringIO(csv_text))
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce") / 100.0
            return df.set_index("date").sort_index()

        url5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        r5 = requests.get(url5, timeout=30)
        z5 = zipfile.ZipFile(io.BytesIO(r5.content))
        with z5.open(z5.namelist()[0]) as f:
            text5 = f.read().decode("latin1")
        df5 = _parse_ff_csv(text5)

        urlm = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        rm = requests.get(urlm, timeout=30)
        zm = zipfile.ZipFile(io.BytesIO(rm.content))
        with zm.open(zm.namelist()[0]) as f:
            textm = f.read().decode("latin1")
        dfm = _parse_ff_csv(textm)
        # MOM CSV has "Mom" as first data column; rename to MOM
        mom_col = [c for c in dfm.columns if "mom" in c.lower()]
        if mom_col:
            dfm = dfm.rename(columns={mom_col[0]: "MOM"})
        elif len(dfm.columns) > 0:
            dfm = dfm.rename(columns={dfm.columns[0]: "MOM"})

        ff = df5.join(dfm[["MOM"]], how="left")
        ff = ff.rename(columns=str.strip)
        ff = ff.rename(columns={
            "Mkt-RF": "FF_MktRF", "SMB": "FF_SMB", "HML": "FF_HML",
            "RMW": "FF_RMW",     "CMA": "FF_CMA",  "RF":  "FF_RF",
        })
        ff.to_parquet(self.ff_daily_pq)
        return ff

    # ── FRED series (direct CSV download, no pandas_datareader) ─────────────
    def fetch_fred(self, series: Iterable[str], refresh: bool = False) -> pd.DataFrame:
        """Download FRED series via public CSV endpoint — no API key required."""
        if self.fred_pq.exists() and not refresh:
            return pd.read_parquet(self.fred_pq)
        if not self.allow_network:
            raise FileNotFoundError("fred.parquet missing and network disabled")

        import requests

        frames = []
        for sid in series:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            try:
                r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"])
                df = df.rename(columns={"DATE": "date", df.columns[1]: sid})
                df = df.set_index("date").sort_index()
                df[sid] = pd.to_numeric(df[sid], errors="coerce")
                frames.append(df[[sid]])
            except Exception as e:
                import warnings
                warnings.warn(f"FRED {sid} fetch failed: {e}")
        if not frames:
            raise RuntimeError("No FRED series fetched")
        fred = pd.concat(frames, axis=1).sort_index()
        fred.to_parquet(self.fred_pq)
        return fred

    # ── WRDS / CRSP ──────────────────────────────────────────────────────────
    #
    # Prerequisites: pip install wrds; run wrds.Connection() once to store
    # credentials in ~/.pgpass. All methods cache to parquet on first call.
    #
    # Impact on strategy:
    #   fetch_crsp_spx_membership → eliminates survivorship bias in universe
    #   fetch_crsp_daily          → accurate adjusted prices, splits, dividends
    #   fetch_ibes_earnings       → earnings blackout gate (biggest MaxDD fix)
    #   fetch_markit_borrow       → screen pairs by borrow cost/availability

    def fetch_crsp_spx_membership(self, refresh: bool = False) -> pd.DataFrame:
        """Historical S&P 500 membership from CRSP (msp500list).

        Returns a DataFrame with columns [permno, ticker, start, end] where
        start/end are the dates a stock entered/left the SPX index. Used to
        build a point-in-time universe that avoids survivorship bias.

        WRDS table: crsp.msp500list
        Columns used: permno, namedt (start), nameendt (end)
        """
        cache = self.data_dir / "crsp_spx_membership.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        import wrds
        conn = wrds.Connection()
        sql = """
            SELECT permno, namedt AS start_date, nameendt AS end_date
            FROM crsp.msp500list
            ORDER BY namedt
        """
        df = conn.raw_sql(sql, date_cols=["start_date", "end_date"])
        conn.close()

        # Join CRSP permnos to tickers via crsp.stocknames
        conn2 = wrds.Connection()
        names_sql = """
            SELECT permno, ticker, comnam,
                   namedt AS name_start, nameendt AS name_end
            FROM crsp.stocknames
        """
        names = conn2.raw_sql(names_sql, date_cols=["name_start", "name_end"])
        conn2.close()

        df = df.merge(names[["permno", "ticker", "name_start", "name_end"]], on="permno", how="left")
        df.to_parquet(cache)
        return df

    def fetch_crsp_daily(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: str = "2014-01-01",
        refresh: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CRSP adjusted daily close prices and volume (dsf table).

        More accurate than yfinance: uses CRSP-standard adjustment factors,
        handles splits and dividends correctly to the day, includes delisted
        returns (critical for survivorship-bias-free backtesting).

        WRDS table: crsp.dsf + crsp.stocknames for ticker mapping.

        Returns:
            close_df:  (dates × tickers) adjusted close prices
            volume_df: (dates × tickers) daily share volume
        """
        close_cache  = self.data_dir / "crsp_close.parquet"
        volume_cache = self.data_dir / "crsp_volume.parquet"
        if close_cache.exists() and volume_cache.exists() and not refresh:
            return pd.read_parquet(close_cache), pd.read_parquet(volume_cache)

        import wrds
        conn = wrds.Connection()

        ticker_filter = ""
        if tickers:
            tickers_sql = ", ".join(f"'{t}'" for t in tickers)
            ticker_filter = f"AND n.ticker IN ({tickers_sql})"

        sql = f"""
            SELECT d.date, n.ticker,
                   d.prc / NULLIF(d.cfacpr, 0)  AS adj_close,
                   d.vol                          AS volume
            FROM crsp.dsf d
            JOIN crsp.stocknames n
              ON d.permno = n.permno
             AND d.date BETWEEN n.namedt AND n.nameendt
            WHERE d.date >= '{start}'
              {ticker_filter}
            ORDER BY d.date, n.ticker
        """
        df = conn.raw_sql(sql, date_cols=["date"])
        conn.close()

        df = df.dropna(subset=["adj_close"]).copy()
        df["adj_close"] = df["adj_close"].abs()  # CRSP uses negative prices for bid quotes

        close_df  = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        volume_df = df.pivot(index="date", columns="ticker", values="volume").sort_index()

        close_df.to_parquet(close_cache)
        volume_df.to_parquet(volume_cache)
        return close_df, volume_df

    # ── WRDS / IBES (earnings calendar) ──────────────────────────────────────

    def fetch_ibes_earnings_dates(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: str = "2014-01-01",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """IBES actual earnings announcement dates per ticker.

        Used to implement the earnings blackout gate (t-1, t, t+1 per leg):
        do not enter new pairs, and exit existing positions 1d before, if
        either leg has an earnings announcement within the window.

        WHY THIS MATTERS FOR MAXDD:
        Earnings-driven gaps (±5-10% in one leg) are the primary source of
        hard-stop losses in pairs trading. The other leg does not gap, so the
        spread spikes to |z|>3 and the position is stopped out at max loss.
        A blackout gate eliminates this by construction.

        WRDS table: ibes.statsum_epsus
        Columns used: ticker, anndats (announcement date), fpi (period indicator)

        Returns:
            DataFrame with columns [ticker, earnings_date] — one row per
            scheduled announcement. Use .groupby('ticker')['earnings_date']
            to build per-ticker sets for fast O(1) lookup in backtest.
        """
        cache = self.data_dir / "ibes_earnings.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        import wrds
        conn = wrds.Connection()

        ticker_filter = ""
        if tickers:
            tickers_sql = ", ".join(f"'{t}'" for t in tickers)
            ticker_filter = f"AND ticker IN ({tickers_sql})"

        sql = f"""
            SELECT DISTINCT ticker, anndats AS earnings_date
            FROM ibes.statsum_epsus
            WHERE anndats >= '{start}'
              AND fpi = '1'   -- quarterly actuals only (fpi=1)
              {ticker_filter}
            ORDER BY ticker, anndats
        """
        df = conn.raw_sql(sql, date_cols=["earnings_date"])
        conn.close()

        df.to_parquet(cache)
        return df

    # ── WRDS / Markit borrow (short availability) ─────────────────────────────

    def fetch_markit_borrow(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: str = "2014-01-01",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Markit Securities Finance daily borrow cost and availability.

        Filters out pairs where shorting one leg is expensive (>200 bps/yr)
        or constrained (low available quantity). Both conditions destroy pair
        PnL: high borrow cost erodes net return; recall risk forces premature
        exit at adverse spread levels.

        WRDS table: sfi.security_detail + sfi.loan_rate  (Markit SFI module)
        Key columns: ticker, date, indicative_fee (annualized bps), quantity_on_loan

        Returns:
            DataFrame (dates × tickers) of daily indicative borrow fee in bps.
            NaN = not available (treat as expensive / hard-to-borrow).
        """
        cache = self.data_dir / "markit_borrow.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        import wrds
        conn = wrds.Connection()

        ticker_filter = ""
        if tickers:
            tickers_sql = ", ".join(f"'{t}'" for t in tickers)
            ticker_filter = f"AND s.ticker IN ({tickers_sql})"

        sql = f"""
            SELECT l.loan_date AS date,
                   s.ticker,
                   l.indicative_fee * 100 AS borrow_bps,
                   l.quantity_on_loan
            FROM sfi.loan_rate l
            JOIN sfi.security_detail s ON l.secid = s.secid
            WHERE l.loan_date >= '{start}'
              AND l.indicative_fee IS NOT NULL
              {ticker_filter}
            ORDER BY l.loan_date, s.ticker
        """
        df = conn.raw_sql(sql, date_cols=["date"])
        conn.close()

        borrow_df = df.pivot(index="date", columns="ticker", values="borrow_bps").sort_index()
        borrow_df.to_parquet(cache)
        return borrow_df

    def fetch_bbg(self):
        raise NotImplementedError("Bloomberg integration stub — replace with your BLP/WAPI calls")

    # ── yfinance fallback for equities and ETFs ─────────────────────────────
    def yf_equities(self, n: int = 300, refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Reuse existing module to stay DRY
        from spx_data import download_spx_universe

        if not refresh and self.spx_close_pq.exists() and self.spx_vol_pq.exists():
            return pd.read_parquet(self.spx_close_pq), pd.read_parquet(self.spx_vol_pq)
        close_df, vol_df = download_spx_universe(n=n, refresh=refresh)
        close_df.to_parquet(self.spx_close_pq)
        vol_df.to_parquet(self.spx_vol_pq)
        return close_df, vol_df

    def yf_factor_etfs(self, refresh: bool = False) -> pd.DataFrame:
        from spx_data import download_factor_etfs

        if not refresh and self.factor_close_pq.exists():
            return pd.read_parquet(self.factor_close_pq)
        fc = download_factor_etfs(refresh=refresh)
        fc.to_parquet(self.factor_close_pq)
        return fc

    # ── Unified entrypoint ──────────────────────────────────────────────────
    def load_all(
        self,
        n: int = 300,
        refresh: bool = False,
        prefer_factors: str = "both",  # both | ff_fred_first | etf_only
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """Return (stock_close, stock_volume, stock_returns, factor_close_or_None, factor_returns).

        factor_close_or_None:
            - FF/FRED path → None (no prices), factor_returns is regression-ready
            - ETF path     → adjusted closes, and factor_returns = pct_change of these closes
        """
        # Equities (prefer CRSP later; yfinance fallback for now)
        close_df, vol_df = self.yf_equities(n=n, refresh=refresh)
        stock_returns = close_df.pct_change()

        if prefer_factors in ("ff_fred_first", "both"):
            ff = self.fetch_ff_daily(refresh=refresh)
            # Example FRED macro block — deltas used as returns for rates/credit
            fred_ids = ["DGS2", "DGS10", "T10Y2Y", "BAMLCC0A0CM", "BAMLH0A0HYM2"]
            try:
                fred = self.fetch_fred(fred_ids, refresh=refresh)
                fred_d = fred.diff().add_prefix("FRED_d")
                # Align and combine
                F = ff.join(fred_d, how="outer")
            except Exception:
                F = ff
            # Align to equities
            factor_returns = F.reindex(stock_returns.index)
            factor_close = None
            if prefer_factors == "both":
                # Join ETF returns as additional regressors (investable proxies)
                try:
                    fc = self.yf_factor_etfs(refresh=refresh)
                    etf_rets = fc.pct_change().reindex(stock_returns.index).add_prefix("ETF_")
                    factor_returns = factor_returns.join(etf_rets, how="outer")
                    factor_close = fc.reindex(stock_returns.index)
                except Exception:
                    pass
        else:  # etf_only
            fc = self.yf_factor_etfs(refresh=refresh)
            factor_returns = fc.pct_change().reindex(stock_returns.index)
            factor_close = fc.reindex(stock_returns.index)

        return close_df, vol_df, stock_returns, factor_close, factor_returns
