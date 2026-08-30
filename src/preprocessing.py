# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class DataCleaner(BaseEstimator, TransformerMixin):
    """Inf ve bozuk satırları temizler."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        return df


class TemporalMissingHandler(BaseEstimator, TransformerMixin):
    """
    Eksik değerleri SADECE geçmişi kullanarak doldurur.
    Forward Fill ve Backward Fill (sadece serinin en başı için).
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        # Önce geçmişten geleceğe taşı
        df = df.ffill()
        # Eğer en başta eksik varsa mecburen ilk geçerli değeri geriye taşı
        df = df.bfill()
        return df


class RollingOutlierClipper(BaseEstimator, TransformerMixin):
    """
    Gelecek veriyi (lookahead) kullanmadan, sadece 90 günlük geçmiş 
    hareketli medyana ve sapmaya göre aykırı değerleri kırpar (clipping).
    """
    
    def __init__(self, window=90, threshold=3.0):
        self.window = window
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Sadece hacim ve makro değişkenler gibi serilerde outlier kırpılır
        # Hedef (Target) ve Fiyat (Open/Close) elenmemelidir
        skip_cols = ["Open", "High", "Low", "Close", "Price"]
        target_cols = [c for c in numeric_cols if c not in skip_cols]
        
        for col in target_cols:
            rolling_median = df[col].rolling(window=self.window, min_periods=1).median()
            rolling_std = df[col].rolling(window=self.window, min_periods=1).std()
            
            upper_bound = rolling_median + (self.threshold * rolling_std)
            lower_bound = rolling_median - (self.threshold * rolling_std)
            
            # Değerleri alt/üst sınırlara hapset (clip)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        return df


class TargetGenerator(BaseEstimator, TransformerMixin):
    """
    T+1 için tahmin hedefini (Yön: 1 Up, 0 Down) oluşturur.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if "Close" not in df.columns and "Price" in df.columns:
            df["Close"] = df["Price"]
            
        # Gelecekteki 1 günlük getiri (Target Leakage Önlemi: shift(-1))
        df["target_return_1d"] = df["Close"].pct_change(1).shift(-1)
        
        # İkili Sınıflandırma: Yön (1 = Yükseliş, 0 = Düşüş veya Nötr)
        df["target_direction"] = (df["target_return_1d"] > 0).astype(int)
        
        # Hedefin NaN olduğu (son gün) satırı kaldır. (Çünkü T+1 bilinmiyor)
        df = df.dropna(subset=["target_return_1d"])
        
        return df


def create_preprocessing_pipeline():
    """Temel Veri Temizleme ve Hedef Oluşturma Pipeline'ı."""
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('temporal_missing', TemporalMissingHandler()),
        ('rolling_clipper', RollingOutlierClipper(window=90, threshold=3.0)),
        ('target_gen', TargetGenerator())
    ])
    return pipeline


def create_feature_engineering_pipeline():
    """Özellik Mühendisliği (Feature Engineering) Pipeline'ı."""
    # features.py dosyasında güncellediğimiz sınıfları çağırır
    from .features import (
        TechnicalFeatures, 
        VolatilityFeatures, 
        MicrostructureFeatures,
        MomentumFeatures, 
        TrendFeatures, 
        CycleFeatures, 
        MacroDerivativesFeatures, 
        ShockDetectorFeatures
    )
    
    pipeline = Pipeline([
        ('technical', TechnicalFeatures()),
        ('volatility', VolatilityFeatures()),
        ('microstructure', MicrostructureFeatures()),
        ('momentum', MomentumFeatures()),
        ('trend', TrendFeatures()),
        ('cycle', CycleFeatures()),
        ('macro_derivs', MacroDerivativesFeatures()),
        ('shock_detector', ShockDetectorFeatures())
    ])
    return pipeline
