# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# AUTO sys.path FIX
# ------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_DATA_REL_PATH = Path("data") / "raw" / "btc_dataset_raw.csv"
_HISTORY_START = "2018-01-01"


# ==================================================================
class BitcoinDataLoader:

    def __init__(self, data_path: Path | None = None) -> None:
        self.project_root: Path = _PROJECT_ROOT
        self.data_path: Path = data_path or (self.project_root / _DATA_REL_PATH)
        self._cached_df: pd.DataFrame | None = None

    def update(self, force_reload: bool = False) -> pd.DataFrame:
        """Tüm kaynaklardan verileri günceller, hizalar ve kaydeder."""
        if not force_reload and self._cached_df is not None:
            return self._cached_df.copy()

        log.info("=== Multi-Source Quantitative Data Ingestion Started ===")

        # 1. Spot OHLCV
        df_btc = self._fetch_btc_spot(_HISTORY_START)
        if df_btc is None or df_btc.empty:
            log.error("Primary BTC price feed failed.")
            return self._load_local()

        # 2. Makro Göstergeler (UUP/DXY, VIX, SPY)
        df_macro = self._fetch_macro(_HISTORY_START)

        # 3. Sentiment: Fear & Greed Index
        df_fng = self._fetch_fear_and_greed()

        # 4. Türev: Binance Futures Funding Rate
        df_funding = self._fetch_binance_funding(_HISTORY_START)

        # 5. Tüm serileri Date indeksinde birleştirme
        merged = df_btc.copy()
        for aux_df in [df_macro, df_fng, df_funding]:
            if aux_df is not None and not aux_df.empty:
                merged = merged.join(aux_df, how="left")

        # Forward fill / Backward fill
        merged = merged.ffill().bfill()
        merged = self._validate(merged)

        if merged is not None and not merged.empty:
            self._safe_save(merged)
            self._cached_df = merged
            log.info("=== Ingestion complete: %d rows, %d columns ===", len(merged), len(merged.columns))
            return merged

        return self._load_local()

    # --------------------------------------------------------------
    # FETCHERS
    # --------------------------------------------------------------

    def _fetch_btc_spot(self, start_date: str) -> pd.DataFrame | None:
        """Binance Spot (Fallback: yfinance)."""
        try:
            log.info("Ingesting BTC Spot (Binance)...")
            url = "https://api.binance.com/api/v3/klines"
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            all_rows = []

            while True:
                params = {"symbol": "BTCUSDT", "interval": "1d", "startTime": start_ts, "limit": 1000}
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break

                chunk = pd.DataFrame(data, columns=[
                    "open_time", "Open", "High", "Low", "Close", "Volume",
                    "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
                ])
                chunk["Date"] = pd.to_datetime(chunk["open_time"], unit="ms").dt.normalize()
                chunk = chunk.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                all_rows.append(chunk)

                if len(data) < 1000:
                    break
                start_ts = int(data[-1][6]) + 1

            if all_rows:
                df = pd.concat(all_rows)
                return df[~df.index.duplicated(keep="last")].sort_index()
        except Exception as e:
            log.warning("Binance spot error: %s. Trying yfinance...", e)

        try:
            raw = yf.download("BTC-USD", start=start_date, interval="1d", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            df = raw[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            return df[~df.index.duplicated(keep="last")].sort_index()
        except Exception as exc:
            log.error("BTC spot feed failed completely: %s", exc)
            return None

    def _fetch_macro(self, start_date: str) -> pd.DataFrame | None:
        """Dolar Endeksi (UUP), VIX ve SPY verilerini çeker."""
        try:
            log.info("Ingesting Macro Indicators (UUP, VIX, SPY)...")
            tickers = {"UUP": "Macro_DXY", "^VIX": "Macro_VIX", "SPY": "Macro_SPY"}
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts = int(datetime.now().timestamp())

            headers = {"User-Agent": "Mozilla/5.0"}
            dfs = []

            for ticker, col_name in tickers.items():
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    res = resp.json().get("chart", {}).get("result", [])
                    if res:
                        timestamps = res[0]["timestamp"]
                        closes = res[0]["indicators"]["quote"][0]["close"]
                        t_df = pd.DataFrame({
                            "Date": pd.to_datetime(timestamps, unit="s").normalize(),
                            col_name: closes
                        })
                        t_df = t_df.dropna().set_index("Date")
                        dfs.append(t_df)
                else:
                    log.warning("Yahoo Finance HTTP %d for %s", resp.status_code, ticker)

            if dfs:
                macro_merged = pd.concat(dfs, axis=1, sort=True)
                return macro_merged[~macro_merged.index.duplicated(keep="last")].sort_index()
            return None
        except Exception as e:
            log.warning("Macro data ingestion skipped: %s", e)
            return None

    def _fetch_fear_and_greed(self) -> pd.DataFrame | None:
        """Alternative.me üzerinden Crypto Fear & Greed endeksini çeker."""
        try:
            log.info("Ingesting Fear & Greed Index...")
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return None

            df = pd.DataFrame(data)
            df["Date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
            df["Sentiment_FnG"] = df["value"].astype(float)
            df = df.set_index("Date")[["Sentiment_FnG"]]
            return df[~df.index.duplicated(keep="last")].sort_index()
        except Exception as e:
            log.warning("Fear & Greed ingestion skipped: %s", e)
            return None

    def _fetch_binance_funding(self, start_date: str) -> pd.DataFrame | None:
        """Binance Futures 8-saatlik fonlama oranlarını çeker."""
        try:
            log.info("Ingesting Binance Futures Funding Rate...")
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            all_records = []

            while True:
                params = {"symbol": "BTCUSDT", "startTime": start_ts, "limit": 1000}
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break

                for row in data:
                    all_records.append({
                        "Date": pd.to_datetime(row["fundingTime"], unit="ms").normalize(),
                        "Funding_Rate": float(row["fundingRate"])
                    })

                if len(data) < 1000:
                    break
                start_ts = int(data[-1]["fundingTime"]) + 1

            if not all_records:
                return None

            df = pd.DataFrame(all_records)
            daily_funding = df.groupby("Date")["Funding_Rate"].mean().to_frame("Deriv_Funding_Avg")
            return daily_funding[~daily_funding.index.duplicated(keep="last")].sort_index()
        except Exception as e:
            log.warning("Funding rate ingestion skipped: %s", e)
            return None

    # --------------------------------------------------------------
    # HELPERS & VALIDATION
    # --------------------------------------------------------------

    def _load_local(self) -> pd.DataFrame:
        if self.data_path.exists():
            df = pd.read_csv(self.data_path, parse_dates=["Date"], index_col="Date")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        return pd.DataFrame()

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            return None
        return df

    def _safe_save(self, df: pd.DataFrame) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", dir=self.data_path.parent, delete=False, encoding="utf-8"
            ) as tmp:
                tmp_path = Path(tmp.name)
                df.to_csv(tmp, index_label="Date")
            shutil.move(str(tmp_path), str(self.data_path))
            log.info("Merged dataset safely saved to: %s", self.data_path)
        except Exception as exc:
            log.error("Save error: %s", exc)
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

