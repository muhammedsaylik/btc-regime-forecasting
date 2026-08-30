# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


class TechnicalFeatures(BaseEstimator, TransformerMixin):
    """Teknik analiz ve temel fiyat hareket özelliklerini hesaplar."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if "Price" not in df.columns and "Close" in df.columns:
            df["Price"] = df["Close"]

        # Logaritmik ve Basit Getiriler
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return_1d"] = df["Close"].pct_change(1)

        # Price Momentum
        df["price_momentum_7d"] = df["Close"].pct_change(7)
        df["price_momentum_30d"] = df["Close"].pct_change(30)

        # High-Low range ve Open-Close Gap
        df["high_low_range"] = (df["High"] - df["Low"]) / (df["Low"] + 1e-10)
        df["open_close_gap"] = (df["Open"] - df["Close"]) / (df["Close"] + 1e-10)

        # Hacim Göstergeleri
        df["volume_change"] = df["Volume"].pct_change()
        df["volume_ma_ratio"] = df["Volume"] / (df["Volume"].rolling(window=20).mean() + 1e-10)

        return df


class VolatilityFeatures(BaseEstimator, TransformerMixin):
    """Volatilite modelleri: Realized, Garman-Klass ve Parkinson."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Rolling Volatilities (Yıllıklandırılmış: Kripto 365 gün aktiftir)
        df["vol_5d"] = df["log_return"].rolling(window=5).std() * np.sqrt(365)
        df["vol_20d"] = df["log_return"].rolling(window=20).std() * np.sqrt(365)
        df["vol_50d"] = df["log_return"].rolling(window=50).std() * np.sqrt(365)

        # Garman-Klass Volatilite
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2
        gk_term = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        df["gk_vol_20d"] = np.sqrt(gk_term.rolling(window=20).mean()) * np.sqrt(365)

        # Parkinson Volatilite
        factor = 1.0 / (4.0 * np.log(2.0))
        df["parkinson_vol_20d"] = np.sqrt(factor * (log_hl.rolling(window=20).mean())) * np.sqrt(365)

        return df


class MicrostructureFeatures(BaseEstimator, TransformerMixin):
    """Mikroyapı ve Likidite özellikleri (Amihud, Spread Proxy, VWAP)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Amihud İlikidite İndeksi
        df["amihud_illiquidity"] = (np.abs(df["log_return"]) / (df["Volume"] + 1e-10)).rolling(window=20).mean()

        # Bid-Ask Spread Proxy
        df["spread_ratio"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-10)

        # Typical Price Deviation Proxy
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["typical_price_dev"] = (df["Close"] - typical_price) / (typical_price + 1e-10)

        return df


class MomentumFeatures(BaseEstimator, TransformerMixin):
    """Klasik Momentum ve Osilatörler (RSI, MACD, ROC)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Wilder's RSI (14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12, 26, 9)
        df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change (ROC 12)
        df["roc_12"] = df["Close"].pct_change(12) * 100.0

        return df


class TrendFeatures(BaseEstimator, TransformerMixin):
    """Trend ve Hareketli Ortalama Mesafeleri."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # SMA & EMA Mesafeleri
        df["sma_50"] = df["Close"].rolling(window=50).mean()
        df["sma_200"] = df["Close"].rolling(window=200).mean()
        df["distance_to_sma_50"] = (df["Close"] - df["sma_50"]) / (df["sma_50"] + 1e-10)
        df["distance_to_sma_200"] = (df["Close"] - df["sma_200"]) / (df["sma_200"] + 1e-10)

        df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["distance_to_ema_50"] = (df["Close"] - df["ema_50"]) / (df["ema_50"] + 1e-10)

        # Hızlı Vektörize Eğim (Slope 20d)
        df["sma_slope_20d"] = (df["Close"] - df["Close"].shift(20)) / 20.0

        return df


class CycleFeatures(BaseEstimator, TransformerMixin):
    """Döngüsel/Takvim Özellikleri (Sin/Cos Dönüşümü)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        idx = pd.to_datetime(df.index)

        df["day_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12.0)

        return df


class MacroDerivativesFeatures(BaseEstimator, TransformerMixin):
    """Loader'dan gelen Makro (DXY, VIX, SPY), Sentiment ve Fonlama verilerini işler."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Sentiment (Fear & Greed) Momentum & Farklar
        if "Sentiment_FnG" in df.columns:
            df["fng_momentum_7d"] = df["Sentiment_FnG"].diff(7)
            df["fng_zscore"] = (df["Sentiment_FnG"] - df["Sentiment_FnG"].rolling(30).mean()) / (
                df["Sentiment_FnG"].rolling(30).std() + 1e-10
            )

        # Fonlama Oranı (Funding Rate) Aşırılıkları (Squeeze Risk)
        if "Deriv_Funding_Avg" in df.columns:
            df["funding_ma_7d"] = df["Deriv_Funding_Avg"].rolling(7).mean()
            df["funding_zscore"] = (df["Deriv_Funding_Avg"] - df["funding_ma_7d"]) / (
                df["Deriv_Funding_Avg"].rolling(30).std() + 1e-10
            )

        # Makro Değişkenler (DXY, VIX, SPY) Getirileri
        if "Macro_DXY" in df.columns:
            df["dxy_return_5d"] = df["Macro_DXY"].pct_change(5)
        if "Macro_VIX" in df.columns:
            df["vix_change_5d"] = df["Macro_VIX"].diff(5)
        if "Macro_SPY" in df.columns:
            df["spy_return_5d"] = df["Macro_SPY"].pct_change(5)

        return df


class ShockDetectorFeatures(BaseEstimator, TransformerMixin):
    """Piyasa Şokları ve Uç Olay Tespitleri."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Crash & Spike Göstergeleri (> %5)
        df["is_crash"] = (df["log_return"] < -0.05).astype(int)
        df["is_spike"] = (df["log_return"] > 0.05).astype(int)

        # Volatilite Patlaması (Z-Score > 2)
        if "vol_20d" not in df.columns:
            df["vol_20d"] = df["log_return"].rolling(window=20).std() * np.sqrt(365)

        vol_ma = df["vol_20d"].rolling(window=50).mean()
        vol_std = df["vol_20d"].rolling(window=50).std()
        df["vol_spike"] = ((df["vol_20d"] - vol_ma) / (vol_std + 1e-10) > 2.0).astype(int)

        # Kümülatif Hacim Deltası (CVD Proxy)
        df["volume_delta"] = np.where(df["Close"] > df["Close"].shift(1), df["Volume"], -df["Volume"])
        df["cvd_20d"] = df["volume_delta"].rolling(window=20).sum()

        return df
        

