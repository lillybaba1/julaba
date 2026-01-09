# Julaba Trading Bot - System Improvements

**Date:** January 8, 2026  
**Version:** 2.0

---

## Overview

This document details all system improvements made to enhance the Julaba trading bot's signal quality, ML capabilities, and AI integration.

---

## 1. Machine Learning Model Enhancement

### Upgraded from v1 to v2

| Aspect | v1 (Before) | v2 (After) |
|--------|-------------|------------|
| **Total Features** | 10 | 22 |
| **Estimators** | 50 | 100 |
| **Max Depth** | 3 | 4 |
| **Learning Rate** | 0.1 | 0.05 |
| **Regularization** | Basic | Subsample (0.8) + Feature subset (sqrt) |

### New Feature Categories (12 additional features)

#### Volume Features (3)
| Feature | Description |
|---------|-------------|
| `volume_ratio` | Current volume vs 20-period average |
| `volume_trend` | Slope of volume moving average (%) |
| `volume_spike` | Max volume in last 5 bars vs average |

#### BTC Correlation Features (3)
| Feature | Description |
|---------|-------------|
| `btc_correlation` | Pearson correlation with BTC returns (-1 to 1) |
| `btc_beta` | Sensitivity to BTC moves (covariance/variance) |
| `btc_relative_strength` | Asset performance vs BTC (%) |

#### Momentum Divergence Features (2)
| Feature | Description |
|---------|-------------|
| `rsi_divergence` | Price vs RSI divergence (bullish/bearish) |
| `momentum_strength` | 10-bar rate of change (%) |

#### Market Microstructure Features (4)
| Feature | Description |
|---------|-------------|
| `volatility_clustering` | Autocorrelation of squared returns |
| `range_expansion` | Current range vs average range |
| `trend_consistency` | % of bars in trend direction |
| `hour_of_day` | Normalized hour (session patterns) |

### Model Persistence
- Model saved to `ml_regime_model.pkl`
- Includes version tracking for migrations
- Old samples discarded on version upgrade (incompatible features)

---

## 2. Signal Generation Enhancement

### New Multi-Factor Confluence System

The `generate_signals()` function now includes:

#### Filter 1: Adaptive ADX Threshold
```python
# Lower threshold when volume confirms
adx_threshold = 20 if volume_ratio > 1.5 else 25
```

#### Filter 2: Volume Confirmation
```python
# Require at least 0.5x average volume
if volume_ratio < 0.5:
    filter_signal()  # Reject low-volume signals
```

#### Filter 3: RSI Extremes Protection
```python
# Don't long when overbought, don't short when oversold
if long_signal and rsi > 75:
    filter_signal()
if short_signal and rsi < 25:
    filter_signal()
```

#### Filter 4: BTC Alignment (High Correlation)
```python
# If LINK correlates >0.7 with BTC:
# - Don't short LINK when BTC is bullish
# - Don't long LINK when BTC is bearish
```

### New Function: `generate_signals_with_details()`

Returns detailed analysis alongside signals:

```python
signals_df, analysis = generate_signals_with_details(df)

# analysis contains:
{
    'signal': 1,  # or -1, 0
    'confluence_score': 4,  # 0-6 scale
    'recommendation': 'STRONG_ENTRY',  # or ENTRY, WEAK_ENTRY, NO_SIGNAL
    'filters_passed': ['Strong trend (ADX=30)', 'High volume (1.5x)'],
    'filters_failed': []
}
```

### Confluence Scoring System

| Score | Recommendation |
|-------|----------------|
| 4+ | STRONG_ENTRY |
| 2-3 | ENTRY |
| 0-1 | WEAK_ENTRY |
| No signal | NO_SIGNAL |

---

## 3. AI Package Upgrade

### Migration from Deprecated to New SDK

| Before | After |
|--------|-------|
| `google-generativeai` | `google-genai` v1.57.0 |
| `genai.GenerativeModel()` | `genai.Client()` |
| `model.generate_content()` | `client.models.generate_content()` |

### New Unified Helper Method

```python
def _generate_content(self, prompt: str) -> Optional[str]:
    """Generate content using the appropriate Gemini API."""
    if GENAI_NEW and self.client:
        # New google-genai SDK
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    elif self.model:
        # Legacy fallback
        response = self.model.generate_content(prompt)
        return response.text
```

### Backward Compatibility
- Graceful fallback to legacy SDK if new SDK unavailable
- Same functionality, no breaking changes

---

## 4. Helper Functions Added

### BTC Data Fetching (with Cache)
```python
def _fetch_btc_data_sync() -> Optional[pd.DataFrame]:
    """Fetch BTC/USDT data with 5-minute cache."""
```

### Volume Feature Calculation
```python
def calculate_volume_features(df: pd.DataFrame) -> Dict[str, float]:
    """Returns volume_ratio, volume_trend, volume_spike."""
```

### BTC Correlation Calculation
```python
def calculate_btc_correlation(df: pd.DataFrame) -> Dict[str, float]:
    """Returns btc_correlation, btc_beta, btc_relative_strength."""
```

### Momentum Divergence Detection
```python
def calculate_momentum_divergence(df: pd.DataFrame) -> Dict[str, float]:
    """Returns rsi_divergence, momentum_strength."""
```

### Market Microstructure Analysis
```python
def calculate_microstructure_features(df: pd.DataFrame) -> Dict[str, float]:
    """Returns volatility_clustering, range_expansion, trend_consistency."""
```

---

## 5. External Connections Verified

| Service | Status | Details |
|---------|--------|---------|
| MEXC Exchange | ✅ Working | LINK/USDT price feed, order execution |
| BTC Data Feed | ✅ Working | For correlation analysis |
| Gemini AI | ✅ Working | New SDK, gemini-2.0-flash-exp model |
| Telegram Bot | ✅ Working | @Julazonbot notifications |

---

## 6. Files Modified

### `indicator.py`
- Added `calculate_volume_features()`
- Added `calculate_btc_correlation()`
- Added `calculate_momentum_divergence()`
- Added `calculate_microstructure_features()`
- Added `_fetch_btc_data_sync()` with caching
- Enhanced `MLRegimeClassifier` (v1 → v2)
- Enhanced `generate_signals()` with multi-factor filtering
- Added `generate_signals_with_details()`

### `ai_filter.py`
- Updated imports for new `google-genai` package
- Added `_generate_content()` helper method
- Updated all AI calls to use new helper
- Added retry logic for robustness
- Maintained backward compatibility with legacy SDK

---

## 7. Configuration

### ML Model Settings
```python
MODEL_VERSION = 2
min_samples_to_train = 50
n_estimators = 100
max_depth = 4
learning_rate = 0.05
subsample = 0.8
max_features = 'sqrt'
```

### Signal Filter Thresholds
```python
ADX_THRESHOLD = 25  # (20 if high volume)
VOLUME_MIN = 0.5    # x average
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 25
BTC_CORRELATION_THRESHOLD = 0.7
```

---

## 8. Usage Examples

### Check ML Model Status
```python
from indicator import MLRegimeClassifier
ml = MLRegimeClassifier()
stats = ml.get_stats()
print(f"Samples: {stats['total_samples']}/50")
print(f"Trained: {stats['is_trained']}")
```

### Get Signal with Confluence Analysis
```python
from indicator import generate_signals_with_details
signals_df, analysis = generate_signals_with_details(df)
if analysis['confluence_score'] >= 4:
    print("Strong entry signal!")
```

### Test AI Connection
```python
from ai_filter import AISignalFilter
ai = AISignalFilter()
response = ai._generate_content("Test")
print(f"AI Working: {response is not None}")
```

---

## 9. Future Enhancements (Roadmap)

- [ ] Add funding rate feature (exchange-specific)
- [ ] Add open interest tracking
- [ ] Implement ensemble of multiple ML models
- [ ] Add walk-forward optimization
- [ ] Add feature importance visualization
- [ ] Implement online learning (continuous training)

---

## 10. Troubleshooting

### ML Model Reset
```bash
# Delete to start fresh
del ml_regime_model.pkl
```

### AI Not Working
```bash
# Check API key
echo %GEMINI_API_KEY%

# Test connection
python -c "from ai_filter import AISignalFilter; print(AISignalFilter().use_ai)"
```

### Low Volume Filtering Too Strict
Adjust in `indicator.py`:
```python
if volume_ratio < 0.3:  # Lower threshold
```

---

*Generated by Julaba Bot System*
