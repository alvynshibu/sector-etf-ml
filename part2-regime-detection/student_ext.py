"""
ECS8051 Coursework 2 - Student Extension Model
Student ID: 40356093
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class StudentExt:
    
    OPTIMAL_K = 2  # Number of regimes
    REGIME_LOOKBACK = 20  # Lookback window for regime features
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.lasso_weight = 0.8
        self.fitted_ = False
        self.pipe_lasso = None
        self.pipe_xgb = None
        self.n_lags = 10
        self.mom_windows = (10, 30)
        self.vol_window = 30
        self.sma_windows = (10, 30)
        self.ema_windows = (10, 30)
        self.rsi_window = 14
        self.feature_names = None
        
        # Regime detection
        self.gmm = None
        self.scaler_regime = None
        self.pca = None
        self.regime_features_fitted_ = False

        

    # Regime Detection Methods
    
    
    def _build_regime_features_single(self, X):

        close = X['Close'] if 'Close' in X.columns else X['close']
        high = X['High'] if 'High' in X.columns else X['high']
        low = X['Low'] if 'Low' in X.columns else X['low']
        volume = X['Volume'] if 'Volume' in X.columns else X['volume']
        
        log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0)
        
        lookback = self.REGIME_LOOKBACK
        features = {}
        
        # Volatility
        features['volatility'] = log_returns.rolling(lookback, min_periods=lookback).std()
        
        # Momentum
        features['momentum'] = log_returns.rolling(lookback, min_periods=lookback).mean()
        
        # Volatility of volatility
        features['vol_of_vol'] = features['volatility'].rolling(lookback, min_periods=lookback).std()
        
        # ATR normalised
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(lookback, min_periods=lookback).mean()
        features['atr_norm'] = atr / close.replace(0, np.nan)
        
        # Volume ratio
        vol_ma = volume.rolling(lookback, min_periods=lookback).mean()
        features['vol_ratio'] = volume / vol_ma.replace(0, np.nan)
        
        # Return-volume correlation
        features['ret_vol_corr'] = log_returns.rolling(lookback, min_periods=lookback).corr(
            volume.pct_change().replace([np.inf, -np.inf], 0.0)
        )
        
        # High-low range normalised
        features['hl_range'] = (high - low) / close.replace(0, np.nan)
        features['hl_range_ma'] = features['hl_range'].rolling(lookback, min_periods=lookback).mean()
        
        df_features = pd.DataFrame(features, index=X.index)
        return df_features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    
    def _fit_regime_model(self, regime_features):

        # Remove any remaining NaN rows
        regime_features_clean = regime_features.dropna()
        
        if len(regime_features_clean) < 100:
            self.regime_features_fitted_ = False
            return None
        
        # Scale features
        self.scaler_regime = StandardScaler()
        X_scaled = self.scaler_regime.fit_transform(regime_features_clean)
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=min(5, X_scaled.shape[1]), random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.OPTIMAL_K,
            covariance_type='full',
            n_init=10,
            random_state=self.random_state
        )
        self.gmm.fit(X_pca)
        
        self.regime_features_fitted_ = True
        return regime_features_clean.index
        
    
    def _get_regime_labels(self, regime_features):

        if not self.regime_features_fitted_:
            # Return neutral regime if not fitted
            return pd.DataFrame({
                'gmm_regime': 0,
                'gmm_prob_0': 0.5,
                'gmm_prob_1': 0.5
            }, index=regime_features.index)
        
        regime_features_clean = regime_features.fillna(0.0)
        X_scaled = self.scaler_regime.transform(regime_features_clean)
        X_pca = self.pca.transform(X_scaled)
        
        labels = self.gmm.predict(X_pca)
        probs = self.gmm.predict_proba(X_pca)
        
        df_regimes = pd.DataFrame({
            'gmm_regime': labels,
            'gmm_prob_0': probs[:, 0],
            'gmm_prob_1': probs[:, 1] if probs.shape[1] > 1 else 1 - probs[:, 0]
        }, index=regime_features_clean.index)
        
        return df_regimes
        
    
    def _add_regime_transition_features(self, df_regimes):

        # Regime change indicator
        df_regimes['regime_change'] = (df_regimes['gmm_regime'].diff() != 0).astype(float)
        df_regimes['regime_change'] = df_regimes['regime_change'].fillna(0)
        
        # Lagged regime probabilities (prevent look-ahead)
        for lag in [1, 2, 3, 5]:
            df_regimes[f'gmm_prob_0_lag{lag}'] = df_regimes['gmm_prob_0'].shift(lag)
            df_regimes[f'gmm_prob_1_lag{lag}'] = df_regimes['gmm_prob_1'].shift(lag)
        
        # Regime momentum
        df_regimes['regime_momentum'] = df_regimes['gmm_prob_0'].diff()
        
        # Days since last regime change
        regime_changes = df_regimes['regime_change'] == 1
        days_since = pd.Series(0, index=df_regimes.index)
        last_change = 0
        for idx_i, (idx, changed) in enumerate(regime_changes.items()):
            if changed:
                last_change = idx_i
            days_since.loc[idx] = idx_i - last_change
        df_regimes['days_since_change'] = days_since
        
        return df_regimes

        

    # Prediction Feature Methods
    
    
    def _log_returns(self, series):

        return np.log(series / series.shift(1)).replace([np.inf, -np.inf], 0.0)

    
    def _rsi_ewm(self, series, window):
        
        diff = series.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (1 - (1 / (1 + rs))).fillna(0.5)
        

    def _make_base_features(self, X):

        close = X['Close'] if 'Close' in X.columns else X['close']
        high = X['High'] if 'High' in X.columns else X['high']
        low = X['Low'] if 'Low' in X.columns else X['low']
        volume = X['Volume'] if 'Volume' in X.columns else X['volume']
        
        lr = self._log_returns(close)
        feats = {}
        
        # Lagged returns
        for i in range(1, self.n_lags + 1):
            feats[f'lag{i}'] = lr.shift(i)
        
        # Momentum
        for w in self.mom_windows:
            feats[f'mom_{w}'] = lr.rolling(w, min_periods=w).mean()
        
        # Volatility
        feats[f'vol_{self.vol_window}'] = lr.rolling(self.vol_window, min_periods=self.vol_window).std()
        
        # SMA distance
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f'sma_dist_{w}'] = (close - sma) / sma.replace(0, np.nan)
        
        # EMA distance
        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            feats[f'ema_dist_{w}'] = (close - ema) / ema.replace(0, np.nan)
        
        # RSI
        feats[f'rsi_{self.rsi_window}'] = self._rsi_ewm(close, self.rsi_window)
        
        # Volume features
        vol_sma = volume.rolling(self.vol_window, min_periods=self.vol_window).mean()
        feats['vol_rel'] = volume / vol_sma.replace(0, np.nan)
        
        # Day range
        feats['day_range'] = (high - low) / close.replace(0, np.nan)
        
        F = pd.DataFrame(feats, index=X.index)
        return F.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        

    def _add_regime_to_features(self, F_base, df_regimes):

        # Select lagged probabilities and transition features (avoid look-ahead)
        regime_cols = [
            c for c in df_regimes.columns
            if (c.startswith('gmm_prob_') and 'lag' in c)
            or c in ['regime_change', 'regime_momentum', 'days_since_change']
        ]
        
        # Align regime features to base feature index
        regime_aligned = df_regimes[regime_cols].reindex(F_base.index)
        
        # One-hot encode lagged hard regime labels
        labels = df_regimes['gmm_regime'].shift(1)  # Lag by 1 to prevent look-ahead
        reg_oh = pd.get_dummies(labels, prefix='reg', dummy_na=False).reindex(F_base.index)
        
        # Combine features
        F_combined = F_base.join(regime_aligned, how='left').join(reg_oh, how='left')
        
        # Interaction features: regime × key base features
        key_feats = [c for c in F_base.columns if c in ['lag1', f'vol_{self.vol_window}']]
        for rf in reg_oh.columns:
            for bf in key_feats:
                if rf in reg_oh.columns and bf in F_base.columns:
                    F_combined[f'{bf}_x_{rf}'] = F_base[bf] * reg_oh[rf].reindex(F_base.index).fillna(0)
        
        return F_combined.fillna(0.0)

    
    # Fit/Predict

    def fit(self, X_train, y_train, meta=None):

        # Build regime features from training data
        regime_features = self._build_regime_features_single(X_train)
        
        # Fit GMM regime model
        self._fit_regime_model(regime_features)
        
        # Get regime labels for training data
        if self.regime_features_fitted_:
            df_regimes = self._get_regime_labels(regime_features)
            df_regimes = self._add_regime_transition_features(df_regimes)
        else:
            df_regimes = None
        
        # Build base prediction features (shifted by 1)
        F_base = self._make_base_features(X_train).shift(1)
        
        # Add regime features
        if df_regimes is not None:
            F_combined = self._add_regime_to_features(F_base, df_regimes)
        else:
            F_combined = F_base
        
        # Align features and target
        y, F = y_train.align(F_combined, join='inner', axis=0)
        F = F.dropna()
        y = y.loc[F.index]
        
        if len(F) < 50:
            self.fitted_ = False
            return self
        
        # Fit Lasso
        self.pipe_lasso = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.1, random_state=self.random_state, max_iter=10000))
        ])
        self.pipe_lasso.fit(F.values, y.values)
        
        # Fit XGBoost
        self.pipe_xgb = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            ))
        ])
        self.pipe_xgb.fit(F.values, y.values)
        
        self.feature_names = F.columns.tolist()
        self.fitted_ = True
        return self
        

    def predict(self, X, meta=None):

        # Build base features
        F_base = self._make_base_features(X)
        
        if not self.fitted_:
            return pd.Series(0.0, index=F_base.index, name='y_pred')
        
        # Build regime features and get labels
        if self.regime_features_fitted_:
            regime_features = self._build_regime_features_single(X)
            df_regimes = self._get_regime_labels(regime_features)
            df_regimes = self._add_regime_transition_features(df_regimes)
            F_combined = self._add_regime_to_features(F_base, df_regimes)
        else:
            F_combined = F_base
        
        # Ensure all training features are present
        for col in self.feature_names:
            if col not in F_combined.columns:
                F_combined[col] = 0.0
        F_combined = F_combined[self.feature_names].fillna(0.0)
        
        # Generate predictions
        y_lasso = self.pipe_lasso.predict(F_combined.values)
        y_xgb = self.pipe_xgb.predict(F_combined.values)
        
        # Weighted ensemble
        y_pred = self.lasso_weight * y_lasso + (1 - self.lasso_weight) * y_xgb
        
        return pd.Series(y_pred, index=F_combined.index, name='y_pred')
