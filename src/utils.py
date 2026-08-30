# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def calculate_vif(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Sabit terim (constant) ekleyerek Variance Inflation Factor (VIF) hesaplar."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_clean = df[numeric_cols].dropna()
    if df_clean.empty or df_clean.shape[1] < 2:
        return pd.DataFrame({"Feature": [], "VIF": []})

    vif_data = []
    try:
        # VIF için sabit terim eklenmesi şarttır
        X_with_const = add_constant(df_clean, has_constant="add")
        cols = [c for c in X_with_const.columns if c != "const"]
        
        for i, col in enumerate(cols):
            # +1 sabit terimin index 0'da olmasından gelir
            val = variance_inflation_factor(X_with_const.values, i + 1)
            vif_data.append({"Feature": col, "VIF": val})
    except Exception as e:
        log.warning("VIF calculation encountered singular matrix: %s", e)
        return pd.DataFrame({"Feature": numeric_cols, "VIF": np.nan})

    res = pd.DataFrame(vif_data)
    return res.sort_values("VIF", ascending=False).reset_index(drop=True)


def test_stationarity(series: pd.Series, name: str = "Series", alpha: float = 0.05) -> dict:
    """ADF ve KPSS durağanlık testlerini uygular."""
    clean_s = series.dropna()
    results = {"Feature": name}

    if len(clean_s) < 20 or clean_s.std() < 1e-10:
        return {
            "Feature": name,
            "ADF_P_Value": 1.0,
            "ADF_Stationary": "No",
            "KPSS_P_Value": 0.0,
            "KPSS_Stationary": "No",
        }

    # 1. ADF Testi (H0: Birim Kök Var / Durağan Değil)
    try:
        adf_res = adfuller(clean_s, autolag="AIC")
        results["ADF_Statistic"] = adf_res[0]
        results["ADF_P_Value"] = adf_res[1]
        results["ADF_Stationary"] = "Yes" if adf_res[1] < alpha else "No"
    except Exception:
        results["ADF_P_Value"] = 1.0
        results["ADF_Stationary"] = "Error"

    # 2. KPSS Testi (H0: Durağandır)
    try:
        kpss_res = kpss(clean_s, regression="c", nlags="auto")
        results["KPSS_Statistic"] = kpss_res[0]
        results["KPSS_P_Value"] = kpss_res[1]
        results["KPSS_Stationary"] = "No" if kpss_res[1] < alpha else "Yes"
    except Exception:
        results["KPSS_P_Value"] = 0.0
        results["KPSS_Stationary"] = "Error"

    return results


def analyze_stationarity_for_features(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Tüm sayısal kolonların durağanlık test özetini çıkarır."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    records = []
    for col in numeric_cols:
        rec = test_stationarity(df[col], name=col)
        records.append(rec)

    return pd.DataFrame(records)


def calculate_correlation_matrix(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Pearson korelasyon matrisi hesaplar."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].corr()


def identify_highly_correlated_features(corr_matrix: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Belirlenen eşik üzerindeki çok yüksek korelasyonlu çiftleri listeler."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                pairs.append({
                    "Feature_1": cols[i],
                    "Feature_2": cols[j],
                    "Correlation": float(val),
                })
    return pd.DataFrame(pairs)


def check_data_leakage_features(feature_names: list[str]) -> pd.DataFrame | None:
    """İsimlendirme tabanlı potansiyel sızıntı denetimi yapar."""
    suspicious = []
    leakage_patterns = {
        "future": ["lead", "future", "tomorrow", "next_"],
        "target_related": ["target", "label", "y_true"],
        "artifacts": ["date_id", "time_id"],
    }
    for feat in feature_names:
        for cat, patterns in leakage_patterns.items():
            for pat in patterns:
                if pat.lower() in feat.lower():
                    suspicious.append({"Feature": feat, "Category": cat, "Pattern": pat})

    return pd.DataFrame(suspicious) if suspicious else None


def analyze_feature_statistics(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Özelliklerin dağılım parametrelerini (Çarpıklık, Basıklık vb.) özetler."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats_list = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 0:
            stats_list.append({
                "Feature": col,
                "Count": len(s),
                "Mean": float(s.mean()),
                "Std": float(s.std()),
                "Min": float(s.min()),
                "Max": float(s.max()),
                "Skewness": float(stats.skew(s)),
                "Kurtosis": float(stats.kurtosis(s)),
                "Missing": int(df[col].isna().sum()),
            })
    return pd.DataFrame(stats_list)