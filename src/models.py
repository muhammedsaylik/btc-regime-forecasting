# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import warnings
from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("joblib").setLevel(logging.ERROR)

log = logging.getLogger(__name__)

# Kütüphane Kontrolleri
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


# ==================================================================
# 1. GAUSSIAN HIDDEN MARKOV MODEL REGIME DETECTOR
# ==================================================================
class GaussianHMMRegimeDetector(BaseEstimator):
    """
    Piyasa rejimlerini (Boğa, Ayı, Yatay) sayısal olarak kararlı
    Gaussian Hidden Markov Modeli ile sınıflandırır.
    """

    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        self.n_states = n_states
        self.random_state = random_state
        self.model: Any = None
        self.scaler = StandardScaler()
        self.state_map_: dict[int, str] = {}

    def fit(self, X: np.ndarray | pd.DataFrame) -> GaussianHMMRegimeDetector:
        if not HMM_AVAILABLE:
            return self

        X_arr = np.asarray(X)
        if len(X_arr) == 0:
            return self

        # Standartlaştırma sayısal taşmaları ve matris tekilliğini engeller
        X_scaled = self.scaler.fit_transform(X_arr)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=300,
            tol=1e-2,
            min_covar=1e-3,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        # Durumları ortalama getiriye göre sırala (0: Bear, 1: Ranging, 2: Bull)
        means = self.model.means_[:, 0]
        sorted_indices = np.argsort(means)

        self.state_map_ = {
            sorted_indices[0]: "bear",
            sorted_indices[1]: "ranging",
            sorted_indices[2]: "bull",
        }
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.ones((len(X), 3)) / 3.0

        X_arr = np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        raw_proba = self.model.predict_proba(X_scaled)

        ordered_proba = np.zeros_like(raw_proba)
        for original_idx, state_name in self.state_map_.items():
            if state_name == "bear":
                ordered_proba[:, 0] = raw_proba[:, original_idx]
            elif state_name == "ranging":
                ordered_proba[:, 1] = raw_proba[:, original_idx]
            elif state_name == "bull":
                ordered_proba[:, 2] = raw_proba[:, original_idx]

        return ordered_proba


# ==================================================================
# 2. FINANCIAL LOSS & WEIGHTING ENGINE
# ==================================================================
class FinancialObjective:
    @staticmethod
    def calculate_weights(
        y_true: np.ndarray,
        sample_returns: np.ndarray | None = None,
        asym_factor: float = 1.5,
    ) -> np.ndarray:
        if sample_returns is None or len(sample_returns) == 0:
            return np.ones(len(y_true), dtype=float)

        abs_ret = np.abs(sample_returns)
        mean_ret = np.mean(abs_ret) + 1e-10
        weights = abs_ret / mean_ret

        downside_mask = (y_true == 0)
        weights[downside_mask] *= asym_factor

        return np.clip(weights, 0.1, 10.0)


# ==================================================================
# 3. DIRECTIONAL FORECASTER WRAPPER
# ==================================================================
class DirectionalForecaster(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type: str = "lightgbm",
        use_financial_loss: bool = True,
        asym_factor: float = 1.5,
        **kwargs: Any,
    ) -> None:
        self.model_type = model_type.lower()
        self.use_financial_loss = use_financial_loss
        self.asym_factor = asym_factor
        self.params = kwargs
        self.model: Any = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            default_lgb = {
                "n_estimators": 150,
                "max_depth": 5,
                "num_leaves": 20,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "random_state": 42,
                "n_jobs": 1,  # Windows subprocess / wmic hatasını önler
                "verbose": -1,
                "objective": "binary",
            }
            default_lgb.update(self.params)
            self.model = LGBMClassifier(**default_lgb)

        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            default_xgb = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 2,
                "random_state": 42,
                "n_jobs": 1,
                "eval_metric": "logloss",
                "objective": "binary:logistic",
            }
            default_xgb.update(self.params)
            self.model = XGBClassifier(**default_xgb)

        else:
            self.model_type = "elasticnet"
            self.model = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                C=0.1,
                solver="saga",
                max_iter=1000,
                random_state=42,
            )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_returns: np.ndarray | None = None,
    ) -> DirectionalForecaster:
        y_arr = np.asarray(y).ravel()
        X_arr = np.asarray(X)

        if self.use_financial_loss and sample_returns is not None:
            weights = FinancialObjective.calculate_weights(
                y_arr, sample_returns, self.asym_factor
            )
        else:
            weights = np.ones(len(y_arr), dtype=float)

        if self.model_type in ["lightgbm", "xgboost", "elasticnet"]:
            self.model.fit(X_arr, y_arr, sample_weight=weights)
        else:
            self.model.fit(X_arr, y_arr)

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X))

    def feature_importance(self) -> np.ndarray | None:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).ravel()
        return None


# ==================================================================
# 4. ADAPTIVE STACKING FINANCIAL ENSEMBLE
# ==================================================================
class AdaptiveEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        use_financial_loss: bool = True,
        threshold_long: float = 0.52,
        threshold_short: float = 0.48,
    ) -> None:
        self.use_financial_loss = use_financial_loss
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.models: dict[str, DirectionalForecaster] = {
            "lgbm": DirectionalForecaster("lightgbm", use_financial_loss=use_financial_loss),
            "xgb": DirectionalForecaster("xgboost", use_financial_loss=use_financial_loss),
            "elasticnet": DirectionalForecaster("elasticnet", use_financial_loss=use_financial_loss),
        }
        self.meta_learner = LogisticRegression(C=1.0, penalty="l2", random_state=42)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_returns: np.ndarray | None = None,
    ) -> AdaptiveEnsemble:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        meta_features = []
        for name, model in self.models.items():
            model.fit(X_arr, y_arr, sample_returns=sample_returns)
            proba_up = model.predict_proba(X_arr)[:, 1].reshape(-1, 1)
            meta_features.append(proba_up)

        S_train = np.hstack(meta_features)
        if self.use_financial_loss and sample_returns is not None:
            weights = FinancialObjective.calculate_weights(y_arr, sample_returns)
            self.meta_learner.fit(S_train, y_arr, sample_weight=weights)
        else:
            self.meta_learner.fit(S_train, y_arr)

        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        meta_features = [
            model.predict_proba(X_arr)[:, 1].reshape(-1, 1)
            for model in self.models.values()
        ]
        S_test = np.hstack(meta_features)
        return self.meta_learner.predict_proba(S_test)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        proba_up = self.predict_proba(X)[:, 1]
        return (proba_up >= 0.50).astype(int)


# ==================================================================
# 5. TEMPORAL WALK-FORWARD VALIDATOR
# ==================================================================
class WalkForwardValidator:
    def __init__(self, n_splits: int = 5, gap: int = 0) -> None:
        self.n_splits = n_splits
        self.gap = gap
        self.scores: list[float] = []
        self.detailed_metrics: list[dict[str, float]] = []

    def split(
        self, X: np.ndarray | pd.DataFrame, y: Any = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = i * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def evaluate(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_returns: np.ndarray | None = None,
    ) -> list[float]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        self.scores = []
        self.detailed_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X_arr, y_arr)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            ret_train = sample_returns[train_idx] if sample_returns is not None else None

            if hasattr(model, "fit") and "sample_returns" in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, sample_returns=ret_train)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5
            loss = log_loss(y_test, proba)
            brier = brier_score_loss(y_test, proba)

            fold_metric = {
                "fold": fold_idx + 1,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_score": f1,
                "roc_auc": roc,
                "log_loss": loss,
                "brier_score": brier,
            }
            self.detailed_metrics.append(fold_metric)
            self.scores.append(roc)

        return self.scores

    @property
    def mean_score(self) -> float | None:
        return float(np.mean(self.scores)) if self.scores else None

    @property
    def std_score(self) -> float | None:
        return float(np.std(self.scores)) if self.scores else None
