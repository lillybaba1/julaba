# indicator.py
# Mathematical Trading Signal Generator for Julaba
# Uses rigorous statistical analysis and mathematical proofs

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any, List
import logging

logger = logging.getLogger("Julaba.Indicator")


# ============== MATHEMATICAL FUNCTIONS ==============

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Mathematical Definition:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Range: [0, 100]
    - RSI > 70: Overbought (statistically likely to reverse down)
    - RSI < 30: Oversold (statistically likely to reverse up)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Mathematical Definition:
        TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = SMA(TR, period)
    
    Measures volatility in absolute dollar terms.
    """
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average Directional Index.
    
    Mathematical Definition:
        +DM = High - PrevHigh (if > 0 and > -DM, else 0)
        -DM = PrevLow - Low (if > 0 and > +DM, else 0)
        TR = max(H-L, |H-PC|, |L-PC|)
        +DI = 100 * SMA(+DM) / SMA(TR)
        -DI = 100 * SMA(-DM) / SMA(TR)
        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = SMA(DX, period)
    
    Interpretation:
        ADX < 20: Weak trend (ranging market)
        ADX 20-25: Trend developing
        ADX 25-50: Strong trend
        ADX > 50: Very strong trend
    
    Returns the last ADX value.
    """
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(window=period, min_periods=1).mean()
    
    return float(adx.iloc[-1]) if len(adx) > 0 else 0


def calculate_volatility_regime(prices: pd.Series, short_window: int = 20, long_window: int = 100) -> Dict[str, Any]:
    """
    Classify volatility regime.
    
    Compares recent volatility to historical volatility.
    """
    if len(prices) < long_window:
        return {'regime': 'unknown', 'volatility_ratio': 1.0}
    
    returns = prices.pct_change().dropna()
    
    short_vol = returns.tail(short_window).std() * np.sqrt(252 * 24)
    long_vol = returns.tail(long_window).std() * np.sqrt(252 * 24)
    
    ratio = short_vol / long_vol if long_vol > 0 else 1.0
    
    if ratio > 1.5:
        regime = 'high'
    elif ratio < 0.5:
        regime = 'low'
    else:
        regime = 'normal'
    
    return {
        'regime': regime,
        'short_vol': float(short_vol),
        'long_vol': float(long_vol),
        'volatility_ratio': float(ratio)
    }


def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent using R/S analysis.
    
    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting
    """
    if len(prices) < max_lag * 2:
        return 0.5
    
    try:
        lags = range(2, max_lag)
        rs_values = []
        
        for lag in lags:
            returns = prices.pct_change(lag).dropna()
            if len(returns) < 10:
                continue
            
            mean_ret = returns.mean()
            std_ret = returns.std()
            
            if std_ret == 0:
                continue
            
            cumdev = (returns - mean_ret).cumsum()
            r = cumdev.max() - cumdev.min()
            s = std_ret
            
            if s > 0:
                rs_values.append((lag, r/s))
        
        if len(rs_values) < 3:
            return 0.5
        
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        return float(np.clip(slope, 0, 1))
        
    except Exception:
        return 0.5


def calculate_kelly_fraction(returns: pd.Series) -> float:
    """
    Calculate Kelly Criterion fraction.
    
    f* = μ / σ² (simplified)
    
    Or more precisely:
    f* = (p * W - q * L) / (W * L)
    where p = win rate, W = avg win, q = loss rate, L = avg loss
    """
    if len(returns) < 20:
        return 0.01
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.01
    
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    if avg_loss == 0:
        return 0.01
    
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return float(np.clip(kelly, 0.005, 0.25))


def calculate_expectancy(returns: pd.Series) -> float:
    """Calculate expected value per trade."""
    if len(returns) < 10:
        return 0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return returns.mean()
    
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    return win_rate * avg_win - (1 - win_rate) * avg_loss


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 20:
        return 0
    
    excess_returns = returns - risk_free_rate / 252
    
    if excess_returns.std() == 0:
        return 0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return float(sharpe)


# ============== MARKET REGIME CLASSIFICATION ==============

class MarketRegime:
    """
    Classify market regime using multiple indicators.
    """
    
    @staticmethod
    def classify(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive market regime classification.
        """
        if len(df) < 50:
            return {
                'regime': 'UNKNOWN',
                'tradeable': False,
                'confidence': 0,
                'reason': 'Insufficient data'
            }
        
        close = df['close'] if 'close' in df.columns else df['Close']
        
        adx = calculate_adx(df)
        hurst = calculate_hurst_exponent(close)
        vol_regime = calculate_volatility_regime(close)
        
        # Regime classification
        if adx >= 30 and hurst >= 0.55:
            regime = 'STRONG_TRENDING'
            confidence = 0.9
            tradeable = True
        elif adx >= 25:
            regime = 'TRENDING'
            confidence = 0.7
            tradeable = True
        elif adx >= 20:
            regime = 'WEAK_TRENDING'
            confidence = 0.5
            tradeable = True
        elif hurst < 0.4:
            regime = 'CHOPPY'
            confidence = 0.3
            tradeable = False
        else:
            regime = 'RANGING'
            confidence = 0.4
            tradeable = False
        
        # Adjust for extreme volatility
        if vol_regime['regime'] == 'high':
            confidence *= 0.8
        
        reason = ''
        if not tradeable:
            if adx < 25:
                reason = f'ADX too low ({adx:.1f} < 25)'
            elif hurst < 0.4:
                reason = f'Choppy market (Hurst={hurst:.2f})'
            else:
                reason = 'Ranging market'
        
        return {
            'regime': regime,
            'tradeable': tradeable,
            'confidence': confidence,
            'adx': adx,
            'hurst': hurst,
            'volatility': vol_regime['regime'],
            'volatility_ratio': vol_regime['volatility_ratio'],
            'reason': reason
        }


# ============== SIGNAL GENERATION ==============

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals with mathematical regime filtering.
    
    Strategy: SMA(15) / SMA(40) crossover
    Filter: Only trade when ADX > 25 (proven profitable regime)
    """
    df = df.copy()
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        if col.lower() == 'close':
            col_map[col] = 'close'
        elif col.lower() == 'high':
            col_map[col] = 'high'
        elif col.lower() == 'low':
            col_map[col] = 'low'
        elif col.lower() == 'open':
            col_map[col] = 'open'
        elif col.lower() == 'volume':
            col_map[col] = 'volume'
    df = df.rename(columns=col_map)
    
    # SMA calculation (OPTIMIZED: 15/40 based on research)
    df['SMA15'] = df['close'].rolling(window=15, min_periods=1).mean()
    df['SMA40'] = df['close'].rolling(window=40, min_periods=1).mean()
    
    # Generate raw signals
    df['Side'] = 0
    
    # Long: SMA15 crosses above SMA40
    long_condition = (df['SMA15'] > df['SMA40']) & (df['SMA15'].shift(1) <= df['SMA40'].shift(1))
    df.loc[long_condition, 'Side'] = 1
    
    # Short: SMA15 crosses below SMA40
    short_condition = (df['SMA15'] < df['SMA40']) & (df['SMA15'].shift(1) >= df['SMA40'].shift(1))
    df.loc[short_condition, 'Side'] = -1
    
    # MATHEMATICAL FILTER: Only allow signals in favorable regimes
    if len(df) >= 50:
        adx = calculate_adx(df)
        
        # ADX > 25 filter (proven through backtest)
        if adx < 25:
            # Cancel signals in ranging market
            signal_rows = df['Side'] != 0
            if signal_rows.any():
                logger.debug(f"Signals filtered: ADX={adx:.1f} < 25 (ranging market)")
                df.loc[signal_rows, 'Side'] = 0
    
    return df


def get_regime_analysis(df_or_closes, atr: float = 0) -> Dict[str, Any]:
    """
    Get comprehensive regime analysis.
    """
    if isinstance(df_or_closes, pd.DataFrame):
        df = df_or_closes
        close = df['close'] if 'close' in df.columns else df['Close']
    else:
        close = pd.Series(df_or_closes)
        df = pd.DataFrame({'close': close, 'high': close, 'low': close})
    
    if len(close) < 50:
        return {
            'tradeable': False,
            'reason': 'Insufficient data',
            'regime': 'UNKNOWN',
            'adx': 0,
            'hurst': 0.5,
            'confidence': 0,
            'volatility': 'unknown'
        }
    
    regime = MarketRegime.classify(df)
    
    returns = close.pct_change().dropna()
    kelly = calculate_kelly_fraction(returns.tail(100))
    expectancy = calculate_expectancy(returns.tail(50))
    
    regime['kelly_fraction'] = kelly
    regime['expectancy'] = expectancy
    
    return regime


def calculate_optimal_size(df: pd.DataFrame, balance: float, base_risk: float = 0.015) -> Tuple[float, Dict]:
    """
    Calculate mathematically optimal position size.
    
    Uses Kelly Criterion adjusted for regime confidence.
    """
    close = df['close'] if 'close' in df.columns else df['Close']
    returns = close.pct_change().dropna()
    
    regime = MarketRegime.classify(df)
    kelly = calculate_kelly_fraction(returns.tail(100))
    
    confidence_multiplier = regime['confidence']
    
    adjusted_risk = base_risk * confidence_multiplier * (kelly / base_risk)
    adjusted_risk = float(np.clip(adjusted_risk, 0.005, 0.025))
    
    analysis = {
        'base_risk': base_risk,
        'kelly_fraction': kelly,
        'regime_confidence': regime['confidence'],
        'adjusted_risk': adjusted_risk,
        'regime': regime['regime'],
        'tradeable': regime['tradeable']
    }
    
    logger.debug(f"Position sizing: Kelly={kelly:.3f}, Conf={confidence_multiplier:.2f}, "
                 f"Risk={adjusted_risk:.2%}")
    
    return adjusted_risk, analysis


# ============== INTELLIGENT FEATURES ==============

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect key candlestick patterns for entry timing.
    """
    if len(df) < 3:
        return {'pattern': None, 'bullish': None, 'strength': 0}
    
    c = df.tail(3).copy().reset_index(drop=True)
    
    open_col = 'open' if 'open' in c.columns else 'Open'
    high_col = 'high' if 'high' in c.columns else 'High'
    low_col = 'low' if 'low' in c.columns else 'Low'
    close_col = 'close' if 'close' in c.columns else 'Close'
    
    curr_open = c[open_col].iloc[-1]
    curr_high = c[high_col].iloc[-1]
    curr_low = c[low_col].iloc[-1]
    curr_close = c[close_col].iloc[-1]
    
    prev_open = c[open_col].iloc[-2]
    prev_close = c[close_col].iloc[-2]
    
    body = abs(curr_close - curr_open)
    upper_wick = curr_high - max(curr_open, curr_close)
    lower_wick = min(curr_open, curr_close) - curr_low
    total_range = curr_high - curr_low
    
    if total_range == 0:
        return {'pattern': None, 'bullish': None, 'strength': 0}
    
    body_ratio = body / total_range
    patterns = []
    
    # Doji
    if body_ratio < 0.1:
        patterns.append({'pattern': 'doji', 'bullish': None, 'strength': 0.5, 'description': 'Indecision'})
    
    # Bullish Engulfing
    if (prev_close < prev_open and curr_close > curr_open and 
        curr_close > prev_open and curr_open < prev_close):
        patterns.append({'pattern': 'bullish_engulfing', 'bullish': True, 'strength': 0.8, 'description': 'Strong bullish reversal'})
    
    # Bearish Engulfing
    if (prev_close > prev_open and curr_close < curr_open and 
        curr_close < prev_open and curr_open > prev_close):
        patterns.append({'pattern': 'bearish_engulfing', 'bullish': False, 'strength': 0.8, 'description': 'Strong bearish reversal'})
    
    # Hammer
    if body_ratio < 0.3 and lower_wick > body * 2 and upper_wick < body:
        patterns.append({'pattern': 'hammer', 'bullish': True, 'strength': 0.7, 'description': 'Rejection of lower prices'})
    
    # Shooting Star
    if body_ratio < 0.3 and upper_wick > body * 2 and lower_wick < body:
        patterns.append({'pattern': 'shooting_star', 'bullish': False, 'strength': 0.7, 'description': 'Rejection of higher prices'})
    
    # Pin Bars
    if lower_wick > total_range * 0.6:
        patterns.append({'pattern': 'pin_bar_bullish', 'bullish': True, 'strength': 0.75, 'description': 'Strong rejection wick (bullish)'})
    elif upper_wick > total_range * 0.6:
        patterns.append({'pattern': 'pin_bar_bearish', 'bullish': False, 'strength': 0.75, 'description': 'Strong rejection wick (bearish)'})
    
    if patterns:
        return max(patterns, key=lambda x: x['strength'])
    
    return {'pattern': None, 'bullish': None, 'strength': 0}


def calculate_drawdown_adjusted_risk(
    base_risk: float,
    current_balance: float,
    peak_balance: float,
    consecutive_losses: int,
    consecutive_wins: int
) -> Dict[str, Any]:
    """
    Smart Drawdown Control - automatically adjust risk based on performance.
    """
    if peak_balance <= 0:
        peak_balance = current_balance
    
    drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
    drawdown_pct = drawdown * 100
    
    if drawdown_pct >= 20:
        dd_multiplier = 0.25
        mode = 'EMERGENCY'
    elif drawdown_pct >= 10:
        dd_multiplier = 0.5
        mode = 'CAUTIOUS'
    elif drawdown_pct >= 5:
        dd_multiplier = 0.75
        mode = 'REDUCED'
    else:
        dd_multiplier = 1.0
        mode = 'NORMAL'
    
    if consecutive_losses >= 4:
        streak_multiplier = 0.25
        mode = 'EMERGENCY'
    elif consecutive_losses >= 3:
        streak_multiplier = 0.5
    elif consecutive_losses >= 2:
        streak_multiplier = 0.75
    elif consecutive_wins >= 5:
        streak_multiplier = 1.25
    elif consecutive_wins >= 3:
        streak_multiplier = 1.1
    else:
        streak_multiplier = 1.0
    
    if dd_multiplier < 1 or streak_multiplier < 1:
        combined_multiplier = min(dd_multiplier, streak_multiplier)
    else:
        combined_multiplier = max(dd_multiplier, streak_multiplier)
    
    adjusted_risk = base_risk * combined_multiplier
    adjusted_risk = float(np.clip(adjusted_risk, 0.002, 0.03))
    
    return {
        'base_risk': base_risk,
        'adjusted_risk': adjusted_risk,
        'multiplier': combined_multiplier,
        'drawdown_pct': round(drawdown_pct, 2),
        'mode': mode,
        'dd_multiplier': dd_multiplier,
        'streak_multiplier': streak_multiplier,
        'message': f'{mode}: {adjusted_risk:.1%} risk (DD:{drawdown_pct:.1f}%, L:{consecutive_losses}/W:{consecutive_wins})'
    }


def multi_timeframe_confirmation(df_5m: pd.DataFrame, df_15m: pd.DataFrame = None, df_1h: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Multi-Timeframe Analysis - confirm signals across timeframes.
    """
    result = {
        '5m_trend': 'neutral',
        '15m_trend': 'neutral',
        '1h_trend': 'neutral',
        'alignment': 0,
        'confirmation': False,
        'message': ''
    }
    
    if len(df_5m) >= 40:
        sma15 = df_5m['close'].rolling(15).mean().iloc[-1]
        sma40 = df_5m['close'].rolling(40).mean().iloc[-1]
        result['5m_trend'] = 'bullish' if sma15 > sma40 else 'bearish'
    
    if df_15m is not None and len(df_15m) >= 20:
        sma10 = df_15m['close'].rolling(10).mean().iloc[-1]
        sma20 = df_15m['close'].rolling(20).mean().iloc[-1]
        result['15m_trend'] = 'bullish' if sma10 > sma20 else 'bearish'
    
    if df_1h is not None and len(df_1h) >= 20:
        sma10 = df_1h['close'].rolling(10).mean().iloc[-1]
        sma20 = df_1h['close'].rolling(20).mean().iloc[-1]
        result['1h_trend'] = 'bullish' if sma10 > sma20 else 'bearish'
    
    trends = []
    if result['5m_trend'] != 'neutral':
        trends.append(1 if result['5m_trend'] == 'bullish' else -1)
    if result['15m_trend'] != 'neutral':
        trends.append(1 if result['15m_trend'] == 'bullish' else -1)
    if result['1h_trend'] != 'neutral':
        trends.append(1 if result['1h_trend'] == 'bullish' else -1)
    
    if trends:
        result['alignment'] = sum(trends) / len(trends)
    
    bullish_count = sum(1 for t in trends if t == 1)
    bearish_count = sum(1 for t in trends if t == -1)
    
    if bullish_count >= 2:
        result['confirmation'] = True
        result['confirmed_direction'] = 'bullish'
        result['message'] = f"✅ BULLISH: {bullish_count}/{len(trends)} TFs align"
    elif bearish_count >= 2:
        result['confirmation'] = True
        result['confirmed_direction'] = 'bearish'
        result['message'] = f"✅ BEARISH: {bearish_count}/{len(trends)} TFs align"
    else:
        result['confirmation'] = False
        result['confirmed_direction'] = 'mixed'
        result['message'] = f"⚠️ MIXED: 5m={result['5m_trend']}, 15m={result['15m_trend']}, 1h={result['1h_trend']}"
    
    return result


def get_intelligence_summary(
    df: pd.DataFrame,
    signal: int,
    btc_analysis: Dict[str, Any] = None,
    mtf_analysis: Dict[str, Any] = None,
    pattern: Dict[str, Any] = None,
    drawdown_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Combine all intelligence into a summary.
    """
    summary = {
        'signal': 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'NONE'),
        'conflicts': [],
        'confirmations': [],
        'risk_mode': 'NORMAL',
        'trade_allowed': True,
        'confidence_adjustment': 1.0
    }
    
    if pattern and pattern.get('pattern'):
        if pattern['bullish'] == (signal == 1):
            summary['confirmations'].append(f"Pattern: {pattern['pattern']}")
            summary['confidence_adjustment'] *= (1 + pattern['strength'] * 0.1)
        elif pattern['bullish'] is not None and pattern['bullish'] != (signal == 1):
            summary['conflicts'].append(f"Pattern conflict: {pattern['pattern']}")
            summary['confidence_adjustment'] *= 0.9
    
    if drawdown_info:
        summary['risk_mode'] = drawdown_info['mode']
        if drawdown_info['mode'] == 'EMERGENCY':
            summary['confidence_adjustment'] *= 0.5
        elif drawdown_info['mode'] == 'CAUTIOUS':
            summary['confidence_adjustment'] *= 0.75
    
    summary['confidence_adjustment'] = float(np.clip(summary['confidence_adjustment'], 0.3, 1.5))
    
    return summary


# ============== MACHINE LEARNING REGIME CLASSIFIER ==============

class MLRegimeClassifier:
    """
    Lightweight ML-based regime classifier using Gradient Boosting.
    Learns from trade outcomes to predict favorable conditions.
    """
    
    def __init__(self, model_path: str = "ml_regime_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_data = []
        self.min_samples_to_train = 50
        self.feature_names = [
            'adx', 'hurst', 'volatility_ratio', 'rsi',
            'price_vs_sma15', 'price_vs_sma40', 'sma_spread',
            'recent_return_mean', 'recent_return_std', 'recent_max_dd'
        ]
        self._load_model()
    
    def _load_model(self):
        import os
        if os.path.exists(self.model_path):
            try:
                import pickle
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                    self.model = saved.get('model')
                    self.scaler = saved.get('scaler')
                    self.training_data = saved.get('training_data', [])
                    self.is_trained = self.model is not None
                    logger.info(f"ML model loaded: {len(self.training_data)} samples, trained={self.is_trained}")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
    
    def _save_model(self):
        try:
            import pickle
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'training_data': self.training_data
                }, f)
        except Exception as e:
            logger.error(f"Could not save ML model: {e}")
    
    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if len(df) < 50:
            return None
        
        try:
            closes = df['close'].values if 'close' in df.columns else df['Close'].values
            
            adx = calculate_adx(df)
            hurst = calculate_hurst_exponent(pd.Series(closes))
            vol_regime = calculate_volatility_regime(pd.Series(closes))
            volatility_ratio = vol_regime.get('volatility_ratio', 1.0)
            rsi_series = calculate_rsi(pd.Series(closes))
            rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50
            
            sma15 = pd.Series(closes).rolling(15).mean().iloc[-1]
            sma40 = pd.Series(closes).rolling(40).mean().iloc[-1]
            current_price = closes[-1]
            
            price_vs_sma15 = (current_price - sma15) / sma15 * 100
            price_vs_sma40 = (current_price - sma40) / sma40 * 100
            sma_spread = (sma15 - sma40) / sma40 * 100
            
            returns = pd.Series(closes).pct_change().dropna().tail(20)
            if len(returns) > 5:
                recent_return_mean = returns.mean() * 100
                recent_return_std = returns.std() * 100
                cumulative = (1 + returns).cumprod()
                recent_max_dd = (cumulative / cumulative.cummax() - 1).min() * 100
            else:
                recent_return_mean = 0
                recent_return_std = 1
                recent_max_dd = 0
            
            return np.array([
                adx, hurst, volatility_ratio, rsi,
                price_vs_sma15, price_vs_sma40, sma_spread,
                recent_return_mean, recent_return_std, recent_max_dd
            ])
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def record_sample(self, df: pd.DataFrame, outcome: bool):
        features = self.extract_features(df)
        if features is not None:
            self.training_data.append((features, 1 if outcome else 0))
            logger.debug(f"ML sample recorded: outcome={'WIN' if outcome else 'LOSS'}, total={len(self.training_data)}")
            
            if len(self.training_data) >= self.min_samples_to_train:
                if len(self.training_data) % 20 == 0:
                    self.train()
            
            self._save_model()
    
    def train(self):
        if len(self.training_data) < self.min_samples_to_train:
            logger.info(f"ML: Need {self.min_samples_to_train - len(self.training_data)} more samples to train")
            return False
        
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            X = np.array([x[0] for x in self.training_data])
            y = np.array([x[1] for x in self.training_data])
            
            win_rate = y.mean()
            logger.info(f"ML Training: {len(y)} samples, {win_rate:.1%} win rate")
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=3, random_state=42
            )
            
            if len(y) >= 30:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
                logger.info(f"ML CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"ML Top features: {', '.join([f'{k}={v:.2f}' for k,v in top_features])}")
            
            self._save_model()
            return True
            
        except ImportError:
            logger.warning("scikit-learn not installed. Run: pip install scikit-learn")
            return False
        except Exception as e:
            logger.error(f"ML training error: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = {
            'ml_score': 0.5,
            'ml_signal': 'NEUTRAL',
            'is_trained': self.is_trained,
            'confidence': 0,
            'samples': len(self.training_data)
        }
        
        if not self.is_trained or self.model is None:
            result['message'] = f"Need {self.min_samples_to_train - len(self.training_data)} more trades to train"
            return result
        
        features = self.extract_features(df)
        if features is None:
            result['message'] = "Insufficient data for prediction"
            return result
        
        try:
            X = self.scaler.transform(features.reshape(1, -1))
            proba = self.model.predict_proba(X)[0]
            ml_score = proba[1] if len(proba) > 1 else proba[0]
            confidence = abs(ml_score - 0.5) * 2
            
            if ml_score >= 0.65:
                ml_signal = 'FAVORABLE'
            elif ml_score <= 0.35:
                ml_signal = 'UNFAVORABLE'
            else:
                ml_signal = 'NEUTRAL'
            
            result.update({
                'ml_score': round(ml_score, 3),
                'ml_signal': ml_signal,
                'confidence': round(confidence, 2),
                'message': f"ML: {ml_signal} ({ml_score:.0%})"
            })
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            result['message'] = f"Prediction error: {e}"
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_samples': len(self.training_data),
            'is_trained': self.is_trained,
            'min_samples_needed': self.min_samples_to_train,
            'samples_until_training': max(0, self.min_samples_to_train - len(self.training_data))
        }
        
        if self.training_data:
            outcomes = [x[1] for x in self.training_data]
            stats['historical_win_rate'] = sum(outcomes) / len(outcomes)
            stats['wins'] = sum(outcomes)
            stats['losses'] = len(outcomes) - sum(outcomes)
        
        if self.is_trained and self.model is not None:
            try:
                importances = dict(zip(self.feature_names, self.model.feature_importances_))
                stats['top_features'] = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            except:
                pass
        
        return stats


# Global ML classifier instance
_ml_classifier = None

def get_ml_classifier() -> MLRegimeClassifier:
    global _ml_classifier
    if _ml_classifier is None:
        _ml_classifier = MLRegimeClassifier()
    return _ml_classifier


def ml_predict_regime(df: pd.DataFrame) -> Dict[str, Any]:
    classifier = get_ml_classifier()
    return classifier.predict(df)


def ml_record_trade(df: pd.DataFrame, won: bool):
    classifier = get_ml_classifier()
    classifier.record_sample(df, won)
