# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestEngine
from .data_loader import BitcoinDataLoader
from .models import (
    AdaptiveEnsemble,
    DirectionalForecaster,
    GaussianHMMRegimeDetector,
    WalkForwardValidator,
)
from .preprocessing import create_feature_engineering_pipeline, create_preprocessing_pipeline

log = logging.getLogger(__name__)


class BTCForecastPipeline:
    """
    Uçtan uca BTC Kantitatif Tahmin, HMM Rejim Filtresi ve Backtest Pipeline'ı.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        use_ensemble: bool = False,
        use_financial_loss: bool = True,
        use_hmm_gatekeeper: bool = True,
        threshold_long: float = 0.53,
        threshold_short: float = 0.00,
    ) -> None:
        self.model_type = model_type.lower()
        self.use_ensemble = use_ensemble
        self.use_hmm_gatekeeper = use_hmm_gatekeeper
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short

        self.loader = BitcoinDataLoader()
        self.feature_pipeline = create_feature_engineering_pipeline()
        self.preprocess_pipeline = create_preprocessing_pipeline()

        if self.use_ensemble:
            self.model = AdaptiveEnsemble(
                use_financial_loss=use_financial_loss,
                threshold_long=threshold_long,
                threshold_short=threshold_short,
            )
        else:
            self.model = DirectionalForecaster(
                model_type=model_type,
                use_financial_loss=use_financial_loss,
            )

        self.validator = WalkForwardValidator(n_splits=5)

    def build_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
        log.info("Building dataset and feature matrix...")
        raw_df = self.loader.update()

        feature_frame = self.feature_pipeline.fit_transform(raw_df)
        processed_frame = self.preprocess_pipeline.fit_transform(feature_frame)
        processed_frame = processed_frame.sort_index()

        target_col = "target_direction"
        return_col = "target_return_1d"

        drop_cols = {
            target_col, return_col, "Open", "High", "Low", "Close", "Price",
            "target", "price_return"
        }
        feature_cols = [c for c in processed_frame.columns if c not in drop_cols]

        X = processed_frame[feature_cols].copy()
        y = processed_frame[target_col].astype(int)
        sample_returns = processed_frame[return_col].values

        # HMM için girdi matrisi: Gerçek üretilen kolonlarla tam eşleştirme
        hmm_candidates = [
            "target_return_1d",
            "log_return",
            "vol_parkinson_20d",
            "vol_parkinson_10d",
            "vol_historical_20d",
            "Funding_Rate",
            "Deriv_Funding_Avg",
            "Sentiment_FnG",
            "Macro_VIX",
            "Macro_DXY"
        ]
        
        hmm_cols = [c for c in hmm_candidates if c in processed_frame.columns]
        
        if not hmm_cols:
            hmm_features = processed_frame[["target_return_1d"]].copy()
        else:
            hmm_features = processed_frame[hmm_cols].copy()

        log.info(
            "Dataset ready: %d samples, %d features (HMM using %d inputs). Target distribution: %s",
            len(processed_frame), len(feature_cols), len(hmm_cols), dict(y.value_counts(normalize=True).round(3))
        )
        return processed_frame, X, y, sample_returns, hmm_features

    def run(self, n_splits: int = 5) -> dict[str, Any]:
        processed_frame, X, y, sample_returns, hmm_features = self.build_dataset()
        self.validator = WalkForwardValidator(n_splits=n_splits)

        log.info("Starting Temporal Walk-Forward Validation (%d folds)...", n_splits)
        scores = self.validator.evaluate(
            model=self.model,
            X=X,
            y=y,
            sample_returns=sample_returns,
        )

        oof_signals = pd.Series(0, index=processed_frame.index, name="signal")
        oof_proba = pd.Series(0.5, index=processed_frame.index, name="proba_up")
        oof_regimes = pd.Series("neutral", index=processed_frame.index, name="hmm_regime")

        for train_idx, test_idx in self.validator.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            ret_train = sample_returns[train_idx]

            # 1. GBDT Modelini Eğit
            if hasattr(self.model, "fit") and "sample_returns" in self.model.fit.__code__.co_varnames:
                self.model.fit(X_train, y_train, sample_returns=ret_train)
            else:
                self.model.fit(X_train, y_train)

            proba = self.model.predict_proba(X_test)[:, 1]
            test_dates = processed_frame.index[test_idx]
            oof_proba.loc[test_dates] = proba

            # 2. HMM Rejim Filtresi Eğitimi ve Uygulanması
            if self.use_hmm_gatekeeper:
                hmm_train = hmm_features.iloc[train_idx].values
                hmm_test = hmm_features.iloc[test_idx].values

                hmm_detector = GaussianHMMRegimeDetector(n_states=3, random_state=42)
                hmm_detector.fit(hmm_train)

                # [P(Bear), P(Ranging), P(Bull)]
                hmm_probs = hmm_detector.predict_proba(hmm_test)
                p_bear = hmm_probs[:, 0]
                p_bull = hmm_probs[:, 2]

                # src/pipeline.py içindeki karar bloğu:
                fold_signal = np.zeros(len(proba), dtype=int)
                for i in range(len(proba)):
                    # Sadece bariz kriz/çöküş rejiminde (P(Bear) > 0.65) Long sinyali engellenir
                    if proba[i] >= self.threshold_long and p_bear[i] < 0.65:
                        fold_signal[i] = 1
                    elif self.threshold_short > 0 and proba[i] <= self.threshold_short and p_bull[i] < 0.30:
                        fold_signal[i] = -1
                    else:
                        fold_signal[i] = 0

                oof_signals.loc[test_dates] = fold_signal
                oof_regimes.loc[test_dates] = np.where(p_bear > 0.40, "Bear", np.where(p_bull > 0.40, "Bull", "Ranging"))
            else:
                fold_signal = np.where(
                    proba >= self.threshold_long, 1,
                    np.where(self.threshold_short > 0 and proba <= self.threshold_short, -1, 0)
                )
                oof_signals.loc[test_dates] = fold_signal

        tested_mask = oof_proba != 0.5
        backtest_df = processed_frame.loc[tested_mask].copy()
        backtest_df["signal"] = oof_signals.loc[tested_mask]
        backtest_df["proba_up"] = oof_proba.loc[tested_mask]
        backtest_df["hmm_regime"] = oof_regimes.loc[tested_mask]

        log.info("Running Quantitative Backtest Engine on HMM-Filtered OOF predictions...")
        engine = BacktestEngine(
            backtest_df,
            signal_col="signal",
            price_col="Close",
            commission_rate=0.0006,
            slippage=0.0002,
        )
        backtest_metrics = engine.run()

        return {
            "dataset": processed_frame,
            "walk_forward_metrics": self.validator.detailed_metrics,
            "scores": scores,
            "mean_score": self.validator.mean_score,
            "std_score": self.validator.std_score,
            "backtest_metrics": backtest_metrics,
            "backtest_df": backtest_df,
        }
