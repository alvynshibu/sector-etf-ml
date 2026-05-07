# Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class Student:
    """
    Implements a fast 80/20 Lasso/XGBoost ensemble.
    Features include advanced technical indicaators.
    The modes uses hardcoded optimal parameters derived from external tuning.
    """

    def __init__(self, random_state: int = 42, H: int = 1):
        self.random_state = random_state
        self.H = H
        self.lasso_weight = 0.8
        self.fitted_ = False
        self.pipe_lasso = None
        self.pipe_xgb = None

        # Feature parameters (tuned to best general performance in notebook)
        self.n_lags = 10
        self.mom_windows = (10, 30)
        self.vol_window = 30
        self.sma_windows = (10, 30)
        self.ema_windows = (10, 30)
        self.rsi_window = 14

    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        return X["Close"]

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        # Calculates log returns and handles errors
        series = pd.Series(series).astype(float)
        return np.log(series / series.shift(1)).replace([np.inf, -np.inf], 0.0)

    @staticmethod
    def _rsi_ewm(series: pd.Series, window: int) -> pd.Series:
        # Calculates ewm based RSI
        diff = series.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 1 - (1 / (1 + rs))
        return rsi.fillna(0.5)

    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates full feature set for model
        """
        
        close = self._close_series(X).astype(float)
        high = X["High"].astype(float)
        low = X["Low"].astype(float)
        volume = X["Volume"].astype(float)
        
        lr = self._log_returns(close)
        
        feats = {}

        # Lags capturing market memory
        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)

        # Momentum and volatility
        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()
        feats[f"vol_{self.vol_window}"] = lr.rolling(self.vol_window, min_periods=self.vol_window).std(ddof=0)

        # SMA/EMA distances measuring how stretched the price is (Mean Reversion)
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f"sma_dist_{w}"] = (close - sma) / sma.replace(0, np.nan)
        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            feats[f"ema_dist_{w}"] = (close - ema) / ema.replace(0, np.nan)

        # RSI
        feats[f"rsi_{self.rsi_window}"] = self._rsi_ewm(close, self.rsi_window)

        # Volume and Rnage
        vol_sma = volume.rolling(self.vol_window, min_periods=self.vol_window).mean()
        feats['vol_rel'] = volume / vol_sma.replace(0, np.nan)
        feats['day_range'] = (high - low) / close.replace(0, np.nan)

        F = pd.DataFrame(feats, index=X.index)
        F = F.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return F

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta: dict = None):
        F = self._make_features(X_train)
        # Lag features by 1 day to ensure causality
        F = F.shift(1)
        y, F = y_train.align(F, join='inner', axis=0)
        F = F.dropna()
        y = y.loc[F.index]

        if F.empty or len(F) < 50:
            self.fitted_ = False
            return self
        
        # Hardcoded parameters based on external tuning
        best_alpha = 0.1  # Lasso winner (low regularisation)
        best_xgb_depth = 3 # XGBoost winner (low complexity/high stability)
        n_est_fixed = 100 

        # Final Lasso Pipeline
        final_lasso_model = Lasso(alpha=best_alpha, random_state=self.random_state, max_iter=10000)
        self.pipe_lasso = Pipeline([('scaler', StandardScaler()), ('model', final_lasso_model)])
        self.pipe_lasso.fit(F.values, y.values)
        
        # Final XGBoost Pipeline
        final_xgb_model = XGBRegressor(n_estimators=n_est_fixed, max_depth=best_xgb_depth, random_state=self.random_state, n_jobs=-1, verbosity=0)
        self.pipe_xgb = Pipeline([('scaler', StandardScaler()), ('model', final_xgb_model)])
        self.pipe_xgb.fit(F.values, y.values)
        
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: dict | None = None) -> pd.Series:
        F = self._make_features(X)

        # Make sure model was trained properly
        if not self.fitted_ or self.pipe_lasso is None or self.pipe_xgb is None:
            return pd.Series(0.0, index=F.index if len(F) else X.index, name='y_pred') 

        F, _ = F.align(X, join='right', axis=0)
        F = F.fillna(0.0)
        
        y_pred_lasso = self.pipe_lasso.predict(F.values)
        y_pred_xgb = self.pipe_xgb.predict(F.values)
        
        # Create the weighted ensemble
        y_pred_blend = (self.lasso_weight * y_pred_lasso) + ((1 - self.lasso_weight) * y_pred_xgb)
        
        return pd.Series(y_pred_blend, index=F.index, name='y_pred')