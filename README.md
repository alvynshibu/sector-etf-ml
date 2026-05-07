# Sector ETF Machine Learning

Two-part machine learning project on six US sector ETFs (XLK, XLP, XLV, XLF, XLE, XLI)
using daily OHLCV data from 2010 to 2024.

## Part 1: Baseline Return Prediction

Feature engineering from raw OHLCV data to predict 1-day log returns:

- Lag returns, momentum, volatility, SMA/EMA distances, RSI, relative volume, intraday range
- Models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- Validation: walk-forward with TimeSeriesSplit to prevent lookahead bias

## Part 2: Regime-Aware Prediction

Extends CW1 by detecting market regimes and using them as predictive features:

- PCA + GMM clustering identifies two regimes: high volatility/crisis (17%) and low volatility/calm (83%)
- Regime features: soft GMM probabilities, transition flags, lagged probabilities, regime momentum
- Walk-forward evaluation with 4-year minimum training window
- Compared against Hallac et al. (2019) Greedy Gaussian Segmentation baseline

## Results

| Model | Mean DirAcc |
|-------|-------------|
| Baseline (CW1) | 51.18% |
| GMM Regime Model (CW2) | 51.53% |
| Hallac et al. GGS | 51.34% |

Regime features improved directional accuracy most strongly for cyclical sectors,
with XLK gaining +2.57 percentage points.

## Structure

part1-prediction/ — baseline feature engineering and model comparison
part2-regime-detection/ — regime detection, feature extension, walk-forward evaluation

## Setup

pip install -r requirements.txt

Data is included in each subdirectory as `prices2.csv`.

## References

Hallac, D., Vare, S., Boyd, S., & Leskovec, J. (2019). Greedy Gaussian Segmentation of Multivariate Time Series. Advances in Data Analysis and Classification, 13(3), 727-751.