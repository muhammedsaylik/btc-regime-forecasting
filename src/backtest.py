# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class BacktestEngine:
    """
    Kripto piyasaları için sermaye korumalı, likidasyon kontrollü
    ve gerçekçi komisyon modelli Backtest Motoru.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "Close",
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0006,  # %0.06 standart taker fee
        slippage: float = 0.0002,         # %0.02 kayma maliyeti
    ) -> None:
        self.df = df.copy()
        self.signal_col = signal_col
        self.price_col = price_col
        self.initial_capital = float(initial_capital)
        self.total_friction = float(commission_rate + slippage)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.price_col not in self.df.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in DataFrame.")
        if self.signal_col not in self.df.columns:
            raise ValueError(f"Signal column '{self.signal_col}' not found in DataFrame.")
        if self.df.empty:
            raise ValueError("Backtest dataframe is empty.")

    @staticmethod
    def _normalize_signal(series: pd.Series) -> pd.Series:
        signal = pd.to_numeric(series, errors="coerce").fillna(0.0)
        # Sadece 1 (Long), -1 (Short) ve 0 (Nötr / Nakit) değerlerini kabul et
        signal = signal.where(signal.isin([-1.0, 0.0, 1.0]), 0.0)
        return signal.astype(float)

    def run(self) -> dict:
        df = self.df.sort_index().copy()
        df[self.signal_col] = self._normalize_signal(df[self.signal_col])
        df[self.price_col] = pd.to_numeric(df[self.price_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[self.price_col, self.signal_col]).copy()

        if len(df) < 2:
            return self._empty_results()

        # 1. Ham Getiri ve Pozisyon Zamanlaması (T+1 Uygulanışı)
        # Sinyal t anında üretilir, pozisyon t+1 kapanışına kadar taşınır
        returns = df[self.price_col].pct_change().fillna(0.0)
        position = df[self.signal_col].shift(1).fillna(0.0)

        # 2. Pozisyon Değişimi ve Komisyon / Slippage Kesintisi
        turnover = position.diff().abs().fillna(position.abs())
        friction_cost = turnover * self.total_friction

        # Günlük Net Strateji Getirisi
        daily_strategy_return = (position * returns) - friction_cost

        # 3. İteratif ve Likidasyon Korumalı Sermaye Eğrisi
        n_days = len(df)
        equity = np.zeros(n_days, dtype=float)
        equity[0] = self.initial_capital

        for i in range(1, n_days):
            current_equity = equity[i - 1] * (1.0 + daily_strategy_return.iloc[i])
            # Sermaye sıfırın altına düşerse likidasyon gerçekleşir (Pozisyonlar kapanır)
            if current_equity <= 0.0:
                equity[i:] = 0.0
                break
            equity[i] = current_equity

        equity_series = pd.Series(equity, index=df.index)
        benchmark_series = self.initial_capital * (1.0 + returns).cumprod()

        # 4. Performans Metrikleri Hesaplaması
        final_value = float(equity_series.iloc[-1])
        total_return = float((final_value / self.initial_capital) - 1.0)
        benchmark_return = float((benchmark_series.iloc[-1] / self.initial_capital) - 1.0)

        # Kripto için 365 gün esası
        years = max(len(df) / 365.0, 1.0 / 365.0)
        if final_value > 0:
            cagr = float((final_value / self.initial_capital) ** (1.0 / years) - 1.0)
        else:
            cagr = -1.0

        # Sharpe & Volatilite (Yıllıklandırılmış: sqrt(365))
        strat_clean_returns = equity_series.pct_change().dropna()
        daily_std = float(strat_clean_returns.std(ddof=0))
        mean_ret = float(strat_clean_returns.mean())
        annualized_vol = daily_std * np.sqrt(365.0)

        sharpe_ratio = float(np.sqrt(365.0) * mean_ret / (daily_std + 1e-10)) if daily_std > 0 else 0.0

        # Drawdown ve Max Drawdown (Asla 1.0 / %100'ü geçemez)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / (rolling_max + 1e-10)
        max_drawdown = float(drawdowns.min())

        # İşlem İstatistikleri
        trade_mask = turnover > 0
        num_trades = int(trade_mask.sum())
        trade_returns = daily_strategy_return[trade_mask]
        win_rate = float((trade_returns > 0).mean()) if len(trade_returns) > 0 else 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe_ratio,
            "annualized_volatility": annualized_vol,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "benchmark_return": benchmark_return,
        }

    def _empty_results(self) -> dict:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "initial_capital": self.initial_capital,
            "final_value": self.initial_capital,
            "benchmark_return": 0.0,
        }


__all__ = ["BacktestEngine"]
