# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path

# Qt/GUI arayüz hatalarını önleyen headless backend ayarı
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Kurumsal Grafik Teması
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def plot_backtest_performance(
    backtest_df: pd.DataFrame,
    output_path: Path = Path("reports/figures/06_equity_curve_and_drawdown.png"),
) -> None:
    """Kümülatif Sermaye Eğrisi ve Drawdown Profilini Çizer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    returns = backtest_df["Close"].pct_change().fillna(0.0)
    pos = backtest_df["signal"].shift(1).fillna(0.0)

    # Sürtünme maliyeti (Komisyon + Slippage: 8 bps)
    pos_change = pos.diff().abs().fillna(0.0)
    friction = pos_change * 0.0008

    strat_ret = pos * returns - friction
    strat_cum = (1.0 + strat_ret).cumprod()
    bench_cum = (1.0 + returns).cumprod()

    # Drawdown Hesaplama
    strat_peaks = strat_cum.cummax()
    strat_dd = (strat_cum - strat_peaks) / strat_peaks
    bench_peaks = bench_cum.cummax()
    bench_dd = (bench_cum - bench_peaks) / bench_peaks

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    # 1. Kümülatif Getiri (Sermaye Eğrisi)
    ax1.plot(
        strat_cum.index,
        strat_cum.values,
        label="Quantitative Strategy (LightGBM + Asymmetric Loss)",
        color="#0066cc",
        lw=2,
    )
    ax1.plot(
        bench_cum.index,
        bench_cum.values,
        label="Benchmark (BTC Buy & Hold)",
        color="#999999",
        ls="--",
        lw=1.5,
        alpha=0.8,
    )
    ax1.set_title(
        "Bitcoin Directional Strategy: Cumulative Out-of-Sample Equity Growth",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Growth Factor ($1.00 Base)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Drawdown Profili
    ax2.fill_between(
        strat_dd.index,
        strat_dd.values * 100,
        0,
        color="#d9534f",
        alpha=0.4,
        label="Strategy Drawdown (%)",
    )
    ax2.plot(
        bench_dd.index,
        bench_dd.values * 100,
        color="#333333",
        lw=1,
        ls=":",
        label="Buy & Hold Drawdown (%)",
    )
    ax2.set_title(
        "Drawdown Profile (Risk & Capital Preservation)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax2.set_xlabel("Timeline", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-100, 5)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log.info("Equity curve and Drawdown plot saved to: %s", output_path)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    output_path: Path = Path("reports/figures/07_feature_importance.png"),
) -> None:
    """Modelin Karar Mekanizmasında En Çok Kullandığı Özellikleri Çizer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        df_imp["feature"],
        df_imp["importance"],
        color="#1f77b4",
        edgecolor="#0e4375",
        height=0.65,
    )

    ax.set_title(
        f"Top {top_n} Features by Model Weight (LightGBM GBDT)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Relative Feature Importance (Gain / Split Count)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    for bar in bars:
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log.info("Feature importance plot saved to: %s", output_path)