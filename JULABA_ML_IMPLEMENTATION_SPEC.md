# Julaba ML Training Data Pipeline — Implementation Spec

**Date:** January 9, 2026  
**For:** Implementation AI (Coder)  
**Priority:** HIGH — Execute this week  
**Approved by:** Heilige, Claude Opus 4.5

---

## Objective

Extract 90 days of historical data from MEXC, run Julaba's signal logic, generate labeled training samples, and bootstrap the ML model so it can start making predictions within 2 weeks instead of 10 months.

---

## Part 1: Data Extraction from MEXC

### 1.1 API Endpoint

```
GET https://api.mexc.com/api/v3/klines
```

### 1.2 Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| symbol | LINKUSDT, ETHUSDT, SOLUSDT | No slash in symbol |
| interval | 15m | Match live strategy timeframe |
| limit | 1000 | Max per request |
| startTime | epoch_ms | 90 days ago |
| endTime | epoch_ms | Now |

### 1.3 Implementation

```python
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

class MEXCDataFetcher:
    BASE_URL = "https://api.mexc.com/api/v3/klines"
    
    def __init__(self):
        self.pairs = ["LINKUSDT", "ETHUSDT", "SOLUSDT"]
        self.interval = "15m"
        self.days_back = 90
    
    def fetch_klines(self, symbol: str, start_time: int, end_time: int) -> list:
        """Fetch klines for a single time window"""
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    
    def fetch_full_history(self, symbol: str) -> pd.DataFrame:
        """Fetch complete 90-day history with pagination"""
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=self.days_back)).timestamp() * 1000)
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            klines = self.fetch_klines(symbol, current_start, end_time)
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Move start to last candle time + 1
            current_start = klines[-1][0] + 1
            
            # Rate limiting
            time.sleep(0.1)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Type conversions
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['symbol'] = symbol
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_all_pairs(self) -> dict:
        """Fetch history for all configured pairs"""
        data = {}
        
        for pair in self.pairs:
            print(f"Fetching {pair}...")
            data[pair] = self.fetch_full_history(pair)
            print(f"  Got {len(data[pair])} candles")
            time.sleep(1)  # Rate limiting between pairs
        
        return data
    
    def save_to_csv(self, data: dict, output_dir: str = "./historical_data"):
        """Save each pair to CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for pair, df in data.items():
            filepath = f"{output_dir}/{pair}_15m_90d.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")


# Execute
if __name__ == "__main__":
    fetcher = MEXCDataFetcher()
    data = fetcher.fetch_all_pairs()
    fetcher.save_to_csv(data)
```

### 1.4 Expected Output

```
historical_data/
├── LINKUSDT_15m_90d.csv   (~8,640 rows)
├── ETHUSDT_15m_90d.csv    (~8,640 rows)
└── SOLUSDT_15m_90d.csv    (~8,640 rows)
```

---

## Part 2: Technical Indicator Calculation

### 2.1 Required Indicators

Calculate these for each candle:

| Indicator | Formula/Library | Parameters |
|-----------|-----------------|------------|
| SMA Fast | `ta.sma(close, 15)` | period=15 |
| SMA Slow | `ta.sma(close, 40)` | period=40 |
| RSI | `ta.rsi(close, 14)` | period=14 |
| ADX | `ta.adx(high, low, close, 14)` | period=14 |
| ATR | `ta.atr(high, low, close, 14)` | period=14 |
| Volume SMA | `ta.sma(volume, 20)` | period=20 |
| Hurst Exponent | Custom implementation | window=100 |

### 2.2 Implementation

```python
import pandas as pd
import numpy as np
import pandas_ta as ta

class IndicatorCalculator:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def calculate_all(self) -> pd.DataFrame:
        """Calculate all required indicators"""
        
        df = self.df
        
        # Moving Averages
        df['sma_15'] = ta.sma(df['close'], length=15)
        df['sma_40'] = ta.sma(df['close'], length=40)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # ADX
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14']
        df['di_plus'] = adx_df['DMP_14']
        df['di_minus'] = adx_df['DMN_14']
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Volume
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # SMA Distance (for ML feature)
        df['sma_distance_percent'] = ((df['close'] - df['sma_40']) / df['sma_40']) * 100
        
        # Hurst Exponent
        df['hurst'] = self._calculate_hurst_rolling(df['close'], window=100)
        
        # Market Regime
        df['regime'] = df.apply(self._classify_regime, axis=1)
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def _calculate_hurst_rolling(self, series: pd.Series, window: int = 100) -> pd.Series:
        """Calculate rolling Hurst exponent"""
        
        def hurst(ts):
            if len(ts) < 20:
                return 0.5
            
            ts = np.array(ts)
            
            # R/S Analysis
            lags = range(2, min(20, len(ts) // 2))
            tau = []
            
            for lag in lags:
                # Calculate standard deviation of lagged differences
                std = np.std(np.subtract(ts[lag:], ts[:-lag]))
                if std > 0:
                    tau.append(std)
                else:
                    tau.append(1e-10)
            
            if len(tau) < 2:
                return 0.5
            
            # Linear fit to log-log plot
            try:
                poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
                return poly[0]
            except:
                return 0.5
        
        return series.rolling(window=window, min_periods=50).apply(hurst, raw=False)
    
    def _classify_regime(self, row) -> str:
        """Classify market regime based on indicators"""
        
        hurst = row.get('hurst', 0.5)
        adx = row.get('adx', 0)
        
        if pd.isna(hurst) or pd.isna(adx):
            return 'UNKNOWN'
        
        if hurst > 0.55 and adx > 25:
            return 'TRENDING'
        elif hurst < 0.45 or adx < 20:
            return 'CHOPPY'
        elif adx > 30:
            return 'VOLATILE'
        else:
            return 'WEAK_TRENDING'


# Execute
if __name__ == "__main__":
    for pair in ["LINKUSDT", "ETHUSDT", "SOLUSDT"]:
        df = pd.read_csv(f"./historical_data/{pair}_15m_90d.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        calculator = IndicatorCalculator(df)
        df_with_indicators = calculator.calculate_all()
        
        df_with_indicators.to_csv(f"./historical_data/{pair}_15m_90d_indicators.csv", index=False)
        print(f"Processed {pair}: {len(df_with_indicators)} rows")
```

---

## Part 3: Signal Generation (Backtest)

### 3.1 Signal Logic

Replicate EXACT logic from live Julaba:

```
LONG Signal:
  - SMA 15 crosses ABOVE SMA 40
  - ADX > 25
  - RSI < 70 (not overbought)
  - Volume > 0.75x average
  - Regime != CHOPPY

SHORT Signal:
  - SMA 15 crosses BELOW SMA 40
  - ADX > 25
  - RSI > 30 (not oversold)
  - Volume > 0.75x average
  - Regime != CHOPPY
```

### 3.2 Implementation

```python
import pandas as pd
import numpy as np

class BacktestSignalGenerator:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.signals = []
        
        # Parameters (MUST match live Julaba)
        self.adx_min = 25
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volume_min_ratio = 0.75
        self.min_bars_between_signals = 8  # 2 hours at 15m
        
        # Position management
        self.atr_multiplier = 2.0
        self.tp1_r = 1.5
        self.tp2_r = 2.5
        self.tp3_r = 4.0
        self.tp1_exit_pct = 0.50
        self.tp2_exit_pct = 0.30
        self.tp3_exit_pct = 0.20
    
    def detect_crossovers(self) -> pd.DataFrame:
        """Detect SMA crossover events"""
        
        df = self.df
        
        # Previous values
        df['sma_15_prev'] = df['sma_15'].shift(1)
        df['sma_40_prev'] = df['sma_40'].shift(1)
        
        # Crossover detection
        df['cross_up'] = (df['sma_15_prev'] <= df['sma_40_prev']) & (df['sma_15'] > df['sma_40'])
        df['cross_down'] = (df['sma_15_prev'] >= df['sma_40_prev']) & (df['sma_15'] < df['sma_40'])
        
        return df
    
    def apply_filters(self, row, direction: str) -> bool:
        """Apply all filters to a potential signal"""
        
        # ADX filter
        if pd.isna(row['adx']) or row['adx'] < self.adx_min:
            return False
        
        # RSI filter
        if direction == 'LONG' and row['rsi'] > self.rsi_overbought:
            return False
        if direction == 'SHORT' and row['rsi'] < self.rsi_oversold:
            return False
        
        # Volume filter
        if pd.isna(row['volume_ratio']) or row['volume_ratio'] < self.volume_min_ratio:
            return False
        
        # Regime filter
        if row['regime'] == 'CHOPPY':
            return False
        
        return True
    
    def calculate_levels(self, entry_price: float, atr: float, direction: str) -> dict:
        """Calculate SL and TP levels"""
        
        stop_distance = atr * self.atr_multiplier
        
        if direction == 'LONG':
            sl = entry_price - stop_distance
            tp1 = entry_price + (stop_distance * self.tp1_r)
            tp2 = entry_price + (stop_distance * self.tp2_r)
            tp3 = entry_price + (stop_distance * self.tp3_r)
        else:  # SHORT
            sl = entry_price + stop_distance
            tp1 = entry_price - (stop_distance * self.tp1_r)
            tp2 = entry_price - (stop_distance * self.tp2_r)
            tp3 = entry_price - (stop_distance * self.tp3_r)
        
        return {
            'stop_loss': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'stop_distance': stop_distance
        }
    
    def simulate_trade_outcome(self, entry_idx: int, direction: str, levels: dict) -> dict:
        """Simulate trade using future price data"""
        
        df = self.df
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row['close']
        
        # Look forward up to 100 bars (~25 hours)
        max_bars = min(100, len(df) - entry_idx - 1)
        
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        sl_hit = False
        exit_bar = None
        exit_price = None
        exit_reason = None
        
        remaining_position = 1.0
        realized_r = 0.0
        
        for i in range(1, max_bars + 1):
            bar = df.iloc[entry_idx + i]
            high = bar['high']
            low = bar['low']
            
            if direction == 'LONG':
                # Check stop loss first
                if low <= levels['stop_loss']:
                    sl_hit = True
                    exit_bar = i
                    exit_price = levels['stop_loss']
                    exit_reason = 'STOP_LOSS'
                    realized_r -= remaining_position * 1.0  # Lose remaining position
                    break
                
                # Check TP levels
                if not tp1_hit and high >= levels['tp1']:
                    tp1_hit = True
                    realized_r += self.tp1_exit_pct * self.tp1_r
                    remaining_position -= self.tp1_exit_pct
                
                if not tp2_hit and high >= levels['tp2']:
                    tp2_hit = True
                    realized_r += self.tp2_exit_pct * self.tp2_r
                    remaining_position -= self.tp2_exit_pct
                
                if not tp3_hit and high >= levels['tp3']:
                    tp3_hit = True
                    realized_r += self.tp3_exit_pct * self.tp3_r
                    remaining_position = 0
                    exit_bar = i
                    exit_price = levels['tp3']
                    exit_reason = 'TP3_FULL'
                    break
            
            else:  # SHORT
                # Check stop loss first
                if high >= levels['stop_loss']:
                    sl_hit = True
                    exit_bar = i
                    exit_price = levels['stop_loss']
                    exit_reason = 'STOP_LOSS'
                    realized_r -= remaining_position * 1.0
                    break
                
                # Check TP levels
                if not tp1_hit and low <= levels['tp1']:
                    tp1_hit = True
                    realized_r += self.tp1_exit_pct * self.tp1_r
                    remaining_position -= self.tp1_exit_pct
                
                if not tp2_hit and low <= levels['tp2']:
                    tp2_hit = True
                    realized_r += self.tp2_exit_pct * self.tp2_r
                    remaining_position -= self.tp2_exit_pct
                
                if not tp3_hit and low <= levels['tp3']:
                    tp3_hit = True
                    realized_r += self.tp3_exit_pct * self.tp3_r
                    remaining_position = 0
                    exit_bar = i
                    exit_price = levels['tp3']
                    exit_reason = 'TP3_FULL'
                    break
        
        # If trade still open after max_bars, close at current price
        if exit_reason is None:
            exit_bar = max_bars
            exit_price = df.iloc[entry_idx + max_bars]['close']
            
            # Calculate remaining P&L
            if direction == 'LONG':
                remaining_r = (exit_price - entry_price) / levels['stop_distance']
            else:
                remaining_r = (entry_price - exit_price) / levels['stop_distance']
            
            realized_r += remaining_position * remaining_r
            exit_reason = 'TIMEOUT'
        
        # Determine win/loss
        outcome = 1 if realized_r > 0 else 0
        
        return {
            'exit_bar': exit_bar,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'tp1_hit': tp1_hit,
            'tp2_hit': tp2_hit,
            'tp3_hit': tp3_hit,
            'sl_hit': sl_hit,
            'realized_r': realized_r,
            'outcome': outcome,
            'duration_minutes': exit_bar * 15
        }
    
    def generate_signals(self) -> list:
        """Generate all signals with outcomes"""
        
        df = self.detect_crossovers()
        last_signal_idx = -self.min_bars_between_signals
        
        signals = []
        
        for idx in range(50, len(df) - 100):  # Skip first 50 (indicator warmup) and last 100 (outcome simulation)
            row = df.iloc[idx]
            
            # Check minimum gap
            if idx - last_signal_idx < self.min_bars_between_signals:
                continue
            
            direction = None
            
            if row['cross_up']:
                direction = 'LONG'
            elif row['cross_down']:
                direction = 'SHORT'
            
            if direction is None:
                continue
            
            # Apply filters
            if not self.apply_filters(row, direction):
                continue
            
            # Calculate levels
            levels = self.calculate_levels(row['close'], row['atr'], direction)
            
            # Simulate outcome
            outcome = self.simulate_trade_outcome(idx, direction, levels)
            
            # Build signal record
            signal = {
                # Identifiers
                'signal_id': f"{row['symbol']}_{idx}",
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'direction': direction,
                
                # Entry conditions (ML features)
                'entry_price': row['close'],
                'entry_atr': row['atr'],
                'entry_atr_percent': row['atr_percent'],
                'entry_rsi': row['rsi'],
                'entry_adx': row['adx'],
                'entry_volume_ratio': row['volume_ratio'],
                'entry_hurst': row['hurst'],
                'entry_sma_distance_percent': row['sma_distance_percent'],
                'entry_regime': row['regime'],
                'entry_hour': row['hour'],
                'entry_day_of_week': row['day_of_week'],
                
                # Levels
                'stop_loss': levels['stop_loss'],
                'tp1': levels['tp1'],
                'tp2': levels['tp2'],
                'tp3': levels['tp3'],
                
                # Outcome
                'exit_price': outcome['exit_price'],
                'exit_reason': outcome['exit_reason'],
                'tp1_hit': outcome['tp1_hit'],
                'tp2_hit': outcome['tp2_hit'],
                'tp3_hit': outcome['tp3_hit'],
                'sl_hit': outcome['sl_hit'],
                'realized_r': outcome['realized_r'],
                'duration_minutes': outcome['duration_minutes'],
                
                # ML Label
                'outcome': outcome['outcome'],
                
                # Source tracking
                'source': 'backtest',
                'sample_weight': 0.3
            }
            
            signals.append(signal)
            last_signal_idx = idx
        
        return signals


# Execute
if __name__ == "__main__":
    all_signals = []
    
    for pair in ["LINKUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"\nProcessing {pair}...")
        
        df = pd.read_csv(f"./historical_data/{pair}_15m_90d_indicators.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['symbol'] = pair
        
        generator = BacktestSignalGenerator(df)
        signals = generator.generate_signals()
        
        print(f"  Generated {len(signals)} signals")
        
        wins = sum(1 for s in signals if s['outcome'] == 1)
        print(f"  Win rate: {wins/len(signals)*100:.1f}%")
        
        all_signals.extend(signals)
    
    # Save combined signals
    signals_df = pd.DataFrame(all_signals)
    signals_df.to_csv("./historical_data/backtest_training_data.csv", index=False)
    print(f"\nTotal signals: {len(all_signals)}")
    print(f"Saved to ./historical_data/backtest_training_data.csv")
```

### 3.3 Expected Output

```
historical_data/
├── LINKUSDT_15m_90d.csv
├── LINKUSDT_15m_90d_indicators.csv
├── ETHUSDT_15m_90d.csv
├── ETHUSDT_15m_90d_indicators.csv
├── SOLUSDT_15m_90d.csv
├── SOLUSDT_15m_90d_indicators.csv
└── backtest_training_data.csv   (~200-300 signals)
```

---

## Part 4: ML Model Training

### 4.1 Model Specification

| Attribute | Value |
|-----------|-------|
| Algorithm | XGBoost Binary Classifier |
| Target | `outcome` (1=win, 0=loss) |
| Features | 11 entry condition features |
| Validation | 80/20 train/test split |
| Metric | AUC-ROC, Accuracy |

### 4.2 Feature List

```python
FEATURE_COLUMNS = [
    'entry_atr_percent',
    'entry_rsi',
    'entry_adx',
    'entry_volume_ratio',
    'entry_hurst',
    'entry_sma_distance_percent',
    'entry_hour',
    'entry_day_of_week',
    'regime_trending',      # One-hot encoded
    'regime_choppy',        # One-hot encoded
    'regime_weak_trending', # One-hot encoded
]
```

### 4.3 Implementation

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

class JulabaMLTrainer:
    
    def __init__(self):
        self.feature_columns = [
            'entry_atr_percent',
            'entry_rsi', 
            'entry_adx',
            'entry_volume_ratio',
            'entry_hurst',
            'entry_sma_distance_percent',
            'entry_hour',
            'entry_day_of_week',
            'regime_trending',
            'regime_choppy',
            'regime_weak_trending',
        ]
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix with one-hot encoding"""
        
        df = df.copy()
        
        # One-hot encode regime
        df['regime_trending'] = (df['entry_regime'] == 'TRENDING').astype(int)
        df['regime_choppy'] = (df['entry_regime'] == 'CHOPPY').astype(int)
        df['regime_weak_trending'] = (df['entry_regime'] == 'WEAK_TRENDING').astype(int)
        
        # Handle missing values
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def train(self, data_path: str) -> dict:
        """Train the model and return metrics"""
        
        # Load data
        df = pd.read_csv(data_path)
        df = self.prepare_features(df)
        
        # Extract features and labels
        X = df[self.feature_columns]
        y = df['outcome']
        weights = df['sample_weight']
        
        # Split data
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        self.model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob),
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'win_rate_actual': y.mean(),
        }
        
        # Feature importance
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        metrics['feature_importance'] = importance
        
        print("\n=== Training Results ===")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"Actual win rate in data: {metrics['win_rate_actual']:.3f}")
        print("\nFeature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
        
        return metrics
    
    def save_model(self, filepath: str = "./models/julaba_ml_v1.json"):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        
        # Also save feature columns for inference
        joblib.dump(self.feature_columns, filepath.replace('.json', '_features.pkl'))
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "./models/julaba_ml_v1.json"):
        """Load trained model"""
        self.model.load_model(filepath)
        self.feature_columns = joblib.load(filepath.replace('.json', '_features.pkl'))
    
    def predict(self, features: dict) -> float:
        """Predict win probability for a single signal"""
        
        # Prepare features
        df = pd.DataFrame([features])
        df = self.prepare_features(df)
        
        X = df[self.feature_columns]
        
        prob = self.model.predict_proba(X)[0][1]
        return prob


# Execute
if __name__ == "__main__":
    trainer = JulabaMLTrainer()
    metrics = trainer.train("./historical_data/backtest_training_data.csv")
    trainer.save_model("./models/julaba_ml_v1.json")
```

---

## Part 5: Integration with Live Bot

### 5.1 ML Predictor Module

```python
# ml_predictor.py

import pandas as pd
import xgboost as xgb
import joblib

class MLPredictor:
    
    def __init__(self, model_path: str = "./models/julaba_ml_v1.json"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.feature_columns = joblib.load(model_path.replace('.json', '_features.pkl'))
        self.min_confidence = 0.50  # Only flag signals below this
    
    def prepare_features(self, signal: dict) -> pd.DataFrame:
        """Convert signal dict to feature DataFrame"""
        
        features = {
            'entry_atr_percent': signal.get('atr_percent', 0),
            'entry_rsi': signal.get('rsi', 50),
            'entry_adx': signal.get('adx', 0),
            'entry_volume_ratio': signal.get('volume_ratio', 1),
            'entry_hurst': signal.get('hurst', 0.5),
            'entry_sma_distance_percent': signal.get('sma_distance_percent', 0),
            'entry_hour': signal.get('hour', 12),
            'entry_day_of_week': signal.get('day_of_week', 0),
            'regime_trending': 1 if signal.get('regime') == 'TRENDING' else 0,
            'regime_choppy': 1 if signal.get('regime') == 'CHOPPY' else 0,
            'regime_weak_trending': 1 if signal.get('regime') == 'WEAK_TRENDING' else 0,
        }
        
        return pd.DataFrame([features])[self.feature_columns]
    
    def predict(self, signal: dict) -> dict:
        """Get ML prediction for a signal"""
        
        X = self.prepare_features(signal)
        
        prob = self.model.predict_proba(X)[0][1]
        
        return {
            'ml_win_probability': prob,
            'ml_confidence': 'HIGH' if prob > 0.6 else 'MEDIUM' if prob > 0.5 else 'LOW',
            'ml_recommendation': 'TAKE' if prob >= self.min_confidence else 'SKIP'
        }


# Integration in main bot
class JulabaBot:
    
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.ml_influence_weight = 0.0  # Start at 0, increase after validation
    
    def evaluate_signal(self, signal: dict) -> dict:
        """Evaluate signal with all filters including ML"""
        
        # Existing filters
        gemini_result = self.gemini_filter.evaluate(signal)
        
        # ML prediction (informational)
        ml_result = self.ml_predictor.predict(signal)
        
        # Log for data collection
        self.log_ml_prediction(signal, ml_result)
        
        # Combined decision
        if self.ml_influence_weight > 0:
            # Weighted combination
            combined_confidence = (
                gemini_result['confidence'] * (1 - self.ml_influence_weight) +
                ml_result['ml_win_probability'] * self.ml_influence_weight
            )
        else:
            combined_confidence = gemini_result['confidence']
        
        return {
            'take_trade': combined_confidence >= 0.70,
            'gemini_confidence': gemini_result['confidence'],
            'ml_win_probability': ml_result['ml_win_probability'],
            'combined_confidence': combined_confidence
        }
```

---

## Part 6: Execution Checklist

### Day 1: Data Extraction

- [ ] Run `MEXCDataFetcher` for LINK, ETH, SOL
- [ ] Verify CSV files created with ~8,640 rows each
- [ ] Spot check data quality (no gaps, correct timestamps)

### Day 2: Indicator Calculation

- [ ] Run `IndicatorCalculator` on all 3 pairs
- [ ] Verify indicator columns added
- [ ] Check for NaN values in first 50 rows (expected, indicator warmup)

### Day 3: Signal Generation

- [ ] Run `BacktestSignalGenerator` on all 3 pairs
- [ ] Verify `backtest_training_data.csv` created
- [ ] Check signal count (expect 200-300 total)
- [ ] Review win rate distribution

### Day 4: Model Training

- [ ] Run `JulabaMLTrainer`
- [ ] Verify accuracy > 55%
- [ ] Verify AUC-ROC > 0.55
- [ ] Save model to `./models/julaba_ml_v1.json`
- [ ] Review feature importance

### Day 5: Integration

- [ ] Add `MLPredictor` module to live bot
- [ ] Set `ml_influence_weight = 0.0`
- [ ] Verify predictions logged but not blocking trades
- [ ] Monitor for errors

### Week 2-4: Validation

- [ ] Collect 50+ real paper trades
- [ ] Compare ML predictions vs actual outcomes
- [ ] If ML accuracy > 55% on real data, increase weight to 0.25
- [ ] Continue monitoring

---

## Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Backtest samples generated | 150 | 250+ |
| ML accuracy (backtest) | 55% | 60%+ |
| ML AUC-ROC (backtest) | 0.55 | 0.62+ |
| Real trades before enabling ML | 50 | 100 |
| ML accuracy (real data) | 53% | 58%+ |

---

## Failure Modes & Mitigations

| Risk | Mitigation |
|------|------------|
| MEXC API rate limit | Add 0.5s delay between requests |
| Insufficient signals generated | Loosen ADX filter to > 20 for backtest only |
| ML overfitting | Use max_depth=4, early stopping |
| Model worse than random | Don't enable ML influence, investigate |
| Real vs backtest distribution shift | Weight real data higher as it accumulates |

---

*Document Version: 1.0*  
*Implementation Deadline: January 16, 2026*
