#!/usr/bin/env python3
"""
Julaba - AI-Enhanced Crypto Trading Bot
Combines the original trading strategy with AI signal filtering and Telegram notifications.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with file output
def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging to both console and file."""
    log_dir = Path(__file__).parent
    log_file = log_dir / "julaba.log"
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler (always DEBUG for full history)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info("="*60)
    logging.info("Log file: %s", log_file)

logger = logging.getLogger("Julaba")

# Import ccxt
try:
    import ccxt.pro as ccxt
except ImportError:
    import ccxt

# Import our modules
from ai_filter import AISignalFilter
from telegram_bot import get_telegram_notifier, TelegramNotifier
from indicator import (
    generate_signals,
    generate_regime_aware_signals,
    smart_btc_filter,
    detect_candlestick_patterns,
    calculate_drawdown_adjusted_risk,
    get_regime_analysis,
    ml_predict_regime,
    ml_record_trade,
    get_ml_classifier,
    calculate_rsi,
    calculate_atr,
    calculate_adx
)

# Import new enhancement modules
from risk_manager import get_risk_manager, RiskManager
from mtf_analyzer import get_mtf_analyzer, MultiTimeframeAnalyzer
from dashboard import get_dashboard
from chart_generator import get_chart_generator
from ml_config import get_ml_config, get_multi_pair_config, MLConfig, MultiPairConfig, TradeLogSchema
from ml_predictor import get_ml_predictor, MLPredictor

# ============== Data Classes ==============

@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    trailing_stop: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    entry_df_snapshot: Optional[pd.DataFrame] = None  # For ML learning
    
    @property
    def remaining_size(self) -> float:
        """Calculate remaining position size."""
        closed = 0.0
        if self.tp1_hit:
            closed += 0.4
        if self.tp2_hit:
            closed += 0.3
        if self.tp3_hit:
            closed += 0.3
        return self.size * (1 - closed)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.remaining_size
        else:
            return (self.entry_price - current_price) * self.remaining_size


@dataclass 
class TradeStats:
    """Track trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    today_pnl: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0))
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


# ============== Main Bot Class ==============

class Julaba:
    """
    AI-Enhanced Trading Bot with Telegram Integration.
    """
    
    # Strategy parameters (from original)
    BASE_TF = "1m"
    AGG_TF_MINUTES = 3
    ATR_PERIOD = 14
    ATR_MULT = 2.0
    RISK_PCT = 0.02  # 2% risk per trade
    # R/R IMPROVED: Targets raised for better risk/reward
    # Old: 1.0/2.0/3.0 -> Required 80% WR for breakeven
    # New: 1.5/2.5/4.0 -> Required ~45% WR for breakeven
    TP1_R = 1.5  # Was 1.0 - now 1.5R for first target
    TP2_R = 2.5  # Was 2.0 - now 2.5R for second target
    TP3_R = 4.0  # Was 3.0 - now 4.0R for runner
    TP1_PCT = 0.4
    TP2_PCT = 0.3
    TP3_PCT = 0.3
    TRAIL_TRIGGER_R = 1.0
    TRAIL_OFFSET_R = 0.5
    WARMUP_BARS = 50  # Matches 150 1m bars → 50 3m bars
    
    # Execution costs (realistic modeling)
    SLIPPAGE_PCT = 0.001  # 0.1% slippage on market orders
    FEE_TAKER = 0.0020    # 0.2% taker fee (MEXC)
    FEE_MAKER = 0.0010    # 0.1% maker fee (MEXC)
    ROUND_TRIP_COST = 0.004  # 0.4% total (2x taker)
    MIN_WIN_R = 1.5  # Minimum win target considering costs
    
    # === REALISTIC EXECUTION PARAMETERS ===
    SLIPPAGE_PCT = 0.001      # 0.1% expected slippage per trade
    MAKER_FEE_PCT = 0.001     # 0.1% maker fee
    TAKER_FEE_PCT = 0.002     # 0.2% taker fee (market orders)
    USE_LIMIT_ORDERS = False  # Use limit orders when possible
    LIMIT_ORDER_TIMEOUT = 30  # Seconds to wait for limit fill before market
    
    # === BTC CORRELATION CRASH PROTECTION ===
    BTC_CRASH_THRESHOLD = -0.03  # -3% BTC move triggers protection
    BTC_CRASH_COOLDOWN = 3600    # 1 hour pause after BTC crash
    BTC_CHECK_INTERVAL = 60      # Check BTC every 60 seconds
    
    # Config file path for persisting settings
    CONFIG_FILE = Path(__file__).parent / "julaba_config.json"
    
    @classmethod
    def _load_persisted_symbol(cls) -> Optional[str]:
        """Load persisted symbol from config file."""
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    symbol = config.get('symbol')
                    if symbol:
                        logger.info(f"📍 Loaded persisted symbol: {symbol}")
                        return symbol
        except Exception as e:
            logger.debug(f"Could not load persisted symbol: {e}")
        return None
    
    def _save_persisted_symbol(self):
        """Save current symbol to config file."""
        try:
            config = {}
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            config['symbol'] = self.SYMBOL
            config['last_updated'] = datetime.now(timezone.utc).isoformat()
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Saved symbol to config: {self.SYMBOL}")
        except Exception as e:
            logger.error(f"Could not save persisted symbol: {e}")
    
    def __init__(
        self,
        paper_balance: Optional[float] = None,
        ai_confidence: float = 0.7,
        log_level: str = "INFO",
        ai_mode: str = "autonomous",  # "filter", "advisory", "autonomous", "hybrid"
        symbol: str = "LINK/USDT",
        scan_interval: int = 60,  # Scan every 60 seconds for more opportunities
        summary_interval: int = 14400  # 4 hours default
    ):
        # Set log level
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # Symbol (now configurable!) - Load persisted symbol if available
        self.SYMBOL = self._load_persisted_symbol() or symbol
        
        # AI Trading Mode
        # "filter" = AI only validates technical signals (default)
        # "advisory" = AI can suggest trades, requires Telegram confirmation
        # "autonomous" = AI can open trades directly with high confidence
        # "hybrid" = AI suggests via Telegram, you confirm
        self.ai_mode = ai_mode
        self.pending_ai_trade = None  # For advisory/hybrid mode confirmation
        self.last_ai_scan_time = None  # Rate limit AI scans
        self.ai_scan_interval = scan_interval  # Configurable scan interval
        self.ai_scan_notify_opportunities_only = True  # Only notify when AI finds opportunity
        self.ai_scan_quiet_interval = 1800  # Notify "no opportunity" every 30 min (if not opportunities_only)
        
        # === AI AUTONOMOUS DECISION TRACKING ===
        self.ai_decision_interval = 300  # Notify every 5 minutes about AI decisions
        self.last_ai_decision_notification = None
        self.ai_decisions_log = []  # Track all AI decisions for transparency
        self.ai_self_adjust_enabled = True  # Allow AI to adjust settings based on performance
        self.last_ai_self_adjust = None  # Rate limit self-adjustments
        self.ai_self_adjust_interval = 1800  # Check every 30 minutes
        
        # Trading state
        self.paper_mode = paper_balance is not None
        self.balance = paper_balance or 10000.0
        self.initial_balance = self.balance
        self.peak_balance = self.balance  # Track peak for drawdown calculation
        self.position: Optional[Position] = None
        self.stats = TradeStats()
        self.start_time = None  # Set when bot actually starts running
        self.paused = False  # Trading pause state
        
        # Streak tracking for intelligent risk management
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Autonomous summary notifications
        self.last_summary_time = None
        self.summary_interval = summary_interval  # Configurable via CLI
        self.last_daily_summary_date = None  # Track daily summary
        self.summary_notifications_enabled = True  # Can be toggled via /summary command
        
        # === DAILY LOSS LIMIT (Circuit Breaker) ===
        self.daily_loss_limit = 0.05  # 5% max daily loss (configurable)
        self.daily_loss_triggered = False  # Circuit breaker state
        self.daily_loss_reset_date = None  # Track when to reset
        
        # === BTC CRASH PROTECTION ===
        self.btc_crash_cooldown = False
        self.btc_crash_cooldown_until = None
        self.btc_crash_threshold = -0.05  # -5% BTC drop
        self.btc_crash_cooldown_minutes = 60  # 1 hour cooldown
        self.last_btc_price = None
        
        # === DRY-RUN MODE ===
        self.dry_run_mode = False  # Log trades without executing
        
        # === REALISTIC EXECUTION TRACKING ===
        self.total_fees_paid = 0.0  # Track all fees for P&L
        self.total_slippage_cost = 0.0  # Track slippage costs
        
        # === BTC CRASH PROTECTION STATE ===
        self.btc_crash_detected = False
        self.btc_crash_until: Optional[datetime] = None
        self.last_btc_price: Optional[float] = None
        self.last_btc_check: Optional[datetime] = None
        self.btc_1h_ago_price: Optional[float] = None  # For 1h change tracking
        
        # === MULTI-SYMBOL SUPPORT (ML Acceleration Plan) ===
        self.ml_config = get_ml_config()
        self.multi_pair_config = get_multi_pair_config()
        
        # Primary symbol (backward compatible)
        self.symbols: List[str] = [symbol]
        
        # Multi-pair state - initialize from config
        enabled_pairs = self.multi_pair_config.get_enabled_pairs()
        self.additional_symbols: List[str] = [p.pair for p in enabled_pairs if p.pair != symbol]
        
        # Track positions per symbol (for multi-pair support)
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.open_position_count: int = 0
        
        # ML data collection (passive - doesn't block trades)
        self.ml_trade_log: List[Dict] = []  # Complete trade logs for ML training
        
        logger.info(f"Multi-pair enabled: {[p.pair for p in enabled_pairs]}")
        logger.info(f"ML Config: influence={self.ml_config.influence_weight}, min_samples={self.ml_config.min_samples_for_predictions}")
        
        # History for Telegram commands
        self.trade_history: List[Dict] = []
        self.signal_history: List[Dict] = []
        self.cached_last_price: float = 0.0
        self.cached_last_ticker: Dict = {}
        self.cached_last_atr: float = 0.0
        
        # Data
        self.bars_1m: List[Dict] = []
        self.bars_agg: pd.DataFrame = pd.DataFrame()
        
        # Exchange
        self.exchange: Optional[ccxt.Exchange] = None
        
        # AI Filter
        self.ai_filter = AISignalFilter(confidence_threshold=ai_confidence)
        
        # === NEW ENHANCEMENT MODULES ===
        # Risk Manager (centralized risk control)
        self.risk_manager = get_risk_manager()
        
        # Multi-Timeframe Analyzer
        self.mtf_analyzer = get_mtf_analyzer()
        
        # Chart Generator
        self.chart_generator = get_chart_generator()
        
        # ML Predictor (XGBoost trained on backtest data)
        self.ml_predictor = get_ml_predictor()
        if self.ml_predictor.is_loaded:
            logger.info(f"🧠 ML Predictor loaded: {self.ml_predictor.metrics.get('accuracy', 0):.1%} accuracy")
        else:
            logger.info("🧠 ML Predictor: Model not loaded (will use when available)")
        
        # Performance Dashboard (optional)
        self.dashboard = get_dashboard(port=5000)
        self.dashboard_enabled = False  # Enable via CLI --dashboard
        
        # Higher timeframe data caching
        self.bars_15m: pd.DataFrame = pd.DataFrame()
        self.bars_1h: pd.DataFrame = pd.DataFrame()
        self.last_htf_update: Optional[datetime] = None
        
        # Equity curve tracking
        self.equity_curve: List[float] = [self.balance]
        
        # Telegram
        self.telegram = get_telegram_notifier()
        self._setup_telegram_callbacks()
        
        # Control
        self.running = False
        
        logger.info(f"Julaba initialized | Paper: {self.paper_mode} | Balance: ${self.balance:,.2f}")
    
    def _setup_telegram_callbacks(self):
        """Setup callbacks for Telegram commands."""
        self.telegram.get_status = self._get_status
        self.telegram.get_positions = self._get_positions
        self.telegram.get_pnl = self._get_pnl
        self.telegram.get_ai_stats = lambda: self.ai_filter.get_stats()
        self.telegram.get_balance = self._get_balance
        self.telegram.get_trades = self._get_trades
        self.telegram.get_market = self._get_market
        self.telegram.get_signals = self._get_signals
        self.telegram.do_stop = self._do_stop
        self.telegram.do_pause = self._do_pause
        self.telegram.do_resume = self._do_resume
        self.telegram.chat_with_ai = self._chat_with_ai
        # AI mode callbacks
        self.telegram.get_ai_mode = lambda: self.ai_mode
        self.telegram.set_ai_mode = self._set_ai_mode
        self.telegram.confirm_ai_trade = self._confirm_ai_trade
        self.telegram.reject_ai_trade = self._reject_ai_trade
        self.telegram.execute_ai_trade = self._execute_ai_trade
        self.telegram.close_ai_trade = self._close_ai_trade
        # Intelligence callbacks
        self.telegram.get_intelligence = self._get_intelligence
        self.telegram.get_ml_stats = self._get_ml_stats
        self.telegram.get_regime = self._get_regime
        # Summary notification toggle
        self.telegram.toggle_summary = self._toggle_summary_notifications
        self.telegram.get_summary_status = lambda: self.summary_notifications_enabled
        # System control for AI
        self.telegram.get_system_params = self._get_system_params
        self.telegram.set_system_param = self._set_system_param
        self.telegram.get_full_system_state = self._get_full_system_state
        # NEW: Enhanced module callbacks
        self.telegram.get_risk_stats = self._get_risk_stats
        self.telegram.get_mtf_analysis = self._get_mtf_analysis
        self.telegram.run_backtest = self._run_backtest
        self.telegram.get_chart = self._get_chart
        self.telegram.get_equity_curve = lambda: self.equity_curve
        
        # Dashboard callbacks
        self.dashboard.get_status = self._get_status
        self.dashboard.get_balance = self._get_balance
        self.dashboard.get_pnl = self._get_pnl
        self.dashboard.get_position = self._get_current_position_dict
        self.dashboard.get_trades = self._get_trades
        self.dashboard.get_regime = self._get_regime
        self.dashboard.get_ai_stats = lambda: self.ai_filter.get_stats()
        self.dashboard.get_equity_curve = lambda: self.equity_curve
        # Enhanced dashboard callbacks
        self.dashboard.get_indicators = self._get_indicators_for_dashboard
        self.dashboard.get_current_signal = self._get_current_signal
        self.dashboard.get_risk_stats = self._get_risk_stats
        self.dashboard.get_mtf_analysis = self._get_mtf_analysis
        self.dashboard.get_params = self._get_system_params
        self.dashboard.get_signals = self._get_signals
        self.dashboard.get_ohlc_data = self._get_ohlc_for_chart
        # NEW: ML status and system logs
        self.dashboard.get_ml_status = self._get_ml_status
        self.dashboard.get_system_logs = self._get_system_logs
        # AI explanation for dashboard info buttons
        self.dashboard.get_ai_explanation = self._get_ai_explanation_for_dashboard
        # Market scanner callbacks
        self.dashboard.get_market_scan = self._get_market_scan_data
        self.dashboard.switch_symbol = self._switch_trading_symbol
        self.dashboard.ai_analyze_markets = self._ai_analyze_all_markets
    
    def _get_ai_explanation_for_dashboard(self, topic: str, display_name: str) -> str:
        """Get AI explanation for a dashboard topic."""
        # Build context with current system state
        context_parts = []
        
        try:
            # Get current values based on topic
            if topic == "current_signal":
                signal = self._get_current_signal()
                context_parts.append(f"Current signal data: {signal}")
            elif topic == "technical_indicators":
                indicators = self._get_indicators_for_dashboard()
                context_parts.append(f"Current indicators: {indicators}")
            elif topic == "market_regime":
                regime = self._get_regime()
                context_parts.append(f"Current regime: {regime}")
            elif topic == "risk_manager":
                risk = self._get_risk_stats()
                context_parts.append(f"Risk stats: {risk}")
            elif topic == "multi_timeframe":
                mtf = self._get_mtf_analysis()
                context_parts.append(f"MTF analysis: {mtf}")
            elif topic == "ai_filter":
                ai_stats = self.ai_filter.get_stats()
                context_parts.append(f"AI stats: {ai_stats}")
                context_parts.append(f"AI mode: {self.ai_mode}")
            elif topic == "current_position":
                pos = self._get_current_position_dict()
                context_parts.append(f"Position: {pos}")
            elif topic == "ml_model":
                ml = self._get_ml_status()
                context_parts.append(f"ML status: {ml}")
            elif topic == "trading_parameters":
                params = self._get_system_params()
                context_parts.append(f"Parameters: {params}")
            elif topic == "live_price_chart":
                context_parts.append(f"Symbol: {self.SYMBOL}")
                context_parts.append(f"Current price: ${self.cached_last_price:.4f}" if self.cached_last_price else "Price: Loading...")
                context_parts.append(f"ATR: {self.cached_last_atr:.4f}" if self.cached_last_atr else "ATR: Loading...")
                regime = self._get_regime()
                context_parts.append(f"Market regime: {regime.get('regime', 'unknown') if regime else 'unknown'}")
            elif topic == "equity_curve":
                context_parts.append(f"Starting balance: ${self.initial_balance:,.2f}")
                context_parts.append(f"Current balance: ${self.balance:,.2f}")
                pnl = self.balance - self.initial_balance
                pnl_pct = (pnl / self.initial_balance) * 100
                context_parts.append(f"Total P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                context_parts.append(f"Equity curve points: {len(self.equity_curve)}")
                context_parts.append(f"Peak balance: ${self.peak_balance:,.2f}")
                drawdown = ((self.peak_balance - self.balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
                context_parts.append(f"Current drawdown: {drawdown:.2f}%")
            
            context = "\\n".join(context_parts)
        except Exception as e:
            context = f"Error getting context: {e}"
        
        # Create prompt for AI
        prompt = f"""You are Julaba's AI assistant explaining a dashboard section to the user.

The user clicked the info button on: **{display_name}**

Current system data:
{context}

Please explain:
1. What this section shows and why it's important
2. What the current values mean
3. Any actionable insights based on current data
4. Tips for using this information in trading decisions

Keep your response concise (under 200 words), friendly, and educational. Use bullet points where helpful.
Format with markdown for readability."""

        try:
            explanation = self.ai_filter._generate_content(prompt)
            return explanation if explanation else "AI explanation temporarily unavailable."
        except Exception as e:
            logger.error(f"AI explanation error: {e}")
            return f"Could not generate explanation: {str(e)}"
    
    # === FULL MARKET SCAN WITH INDICATORS ===
    SCAN_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT", 
                  "ARBUSDT", "SUIUSDT", "APTUSDT", "OPUSDT", "NEARUSDT",
                  "INJUSDT", "TIAUSDT", "SEIUSDT", "WLDUSDT"]
    
    # Autonomous pair switch settings
    AUTO_SWITCH_ENABLED = True
    AUTO_SWITCH_INTERVAL = 900  # Check every 15 minutes
    AUTO_SWITCH_MIN_SCORE_DIFF = 15  # Minimum score difference to switch
    _last_auto_switch_check = 0
    _full_scan_cache = {"data": None, "timestamp": 0}
    FULL_SCAN_CACHE_DURATION = 120  # Cache for 2 minutes

    def _get_market_scan_data_full(self) -> Dict[str, Any]:
        """Get multi-pair market data WITH full indicator calculations."""
        import time
        current_time = time.time()
        
        # Return cached data if still valid
        if (type(self)._full_scan_cache["data"] and 
            current_time - type(self)._full_scan_cache["timestamp"] < type(self).FULL_SCAN_CACHE_DURATION):
            cached = type(self)._full_scan_cache["data"].copy()
            cached["cached"] = True
            cached["cache_age"] = int(current_time - type(self)._full_scan_cache["timestamp"])
            return cached
        
        pairs_data = []
        
        try:
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.mexc({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
            
            for symbol in type(self).SCAN_PAIRS:
                try:
                    # Fetch OHLCV data (100 bars of 15m = 25 hours of data)
                    ohlcv = sync_exchange.fetch_ohlcv(symbol, '15m', limit=100)
                    
                    if len(ohlcv) < 50:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Calculate indicators
                    rsi = calculate_rsi(df['close'], period=14)
                    current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50
                    
                    adx = calculate_adx(df, period=14)
                    
                    # MACD
                    ema12 = df['close'].ewm(span=12).mean()
                    ema26 = df['close'].ewm(span=26).mean()
                    macd = ema12 - ema26
                    macd_signal = macd.ewm(span=9).mean()
                    macd_hist = float(macd.iloc[-1] - macd_signal.iloc[-1])
                    macd_bullish = macd_hist > 0
                    
                    # Volatility (ATR-based)
                    atr = calculate_atr(df, period=14)
                    current_atr = float(atr.iloc[-1]) if len(atr) > 0 else 0
                    current_price = float(df['close'].iloc[-1])
                    atr_pct = (current_atr / current_price * 100) if current_price > 0 else 0
                    
                    # Price momentum (% change over last 6 bars = 1.5 hours)
                    momentum = ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0
                    
                    # Volume trend
                    vol_sma = df['volume'].rolling(20).mean()
                    volume_ratio = float(df['volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1
                    
                    # SMA trend (15/40 as per our strategy)
                    sma15 = df['close'].rolling(15).mean()
                    sma40 = df['close'].rolling(40).mean()
                    trend = "bullish" if sma15.iloc[-1] > sma40.iloc[-1] else "bearish"
                    trend_strength = abs((sma15.iloc[-1] - sma40.iloc[-1]) / sma40.iloc[-1] * 100) if sma40.iloc[-1] > 0 else 0
                    
                    # Signal detection
                    signal = 0
                    signal_text = "none"
                    # Check for recent crossover (last 3 bars)
                    for i in range(-3, 0):
                        if sma15.iloc[i] > sma40.iloc[i] and sma15.iloc[i-1] <= sma40.iloc[i-1]:
                            signal = 1
                            signal_text = "LONG"
                            break
                        elif sma15.iloc[i] < sma40.iloc[i] and sma15.iloc[i-1] >= sma40.iloc[i-1]:
                            signal = -1
                            signal_text = "SHORT"
                            break
                    
                    # Calculate tradability score (0-100)
                    score = self._calculate_pair_score(
                        rsi=current_rsi, adx=adx, atr_pct=atr_pct,
                        volume_ratio=volume_ratio, macd_bullish=macd_bullish,
                        trend_strength=trend_strength, signal=signal
                    )
                    
                    # 24h change from ticker
                    ticker = sync_exchange.fetch_ticker(symbol)
                    change_24h = ticker.get('percentage', 0) or 0
                    volume_24h = ticker.get('quoteVolume', 0) or 0
                    
                    pairs_data.append({
                        "symbol": symbol,
                        "price": current_price,
                        "change": change_24h,
                        "volatility": atr_pct,
                        "volume": volume_24h,
                        "high": float(df['high'].max()),
                        "low": float(df['low'].min()),
                        # New indicator fields
                        "rsi": round(current_rsi, 1),
                        "adx": round(adx, 1),
                        "macd_bullish": macd_bullish,
                        "trend": trend,
                        "trend_strength": round(trend_strength, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "momentum": round(momentum, 2),
                        "signal": signal,
                        "signal_text": signal_text,
                        "score": round(score, 1)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort by score (highest first)
            pairs_data.sort(key=lambda x: x.get('score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Full market scan error: {e}")
        
        result = {
            "pairs": pairs_data,
            "current_symbol": self.SYMBOL,
            "best_pair": pairs_data[0] if pairs_data else None,
            "cached": False,
            "timestamp": current_time
        }
        
        # Cache the result
        type(self)._full_scan_cache = {"data": result, "timestamp": current_time}
        
        return result

    def _calculate_pair_score(self, rsi: float, adx: float, atr_pct: float,
                              volume_ratio: float, macd_bullish: bool,
                              trend_strength: float, signal: int) -> float:
        """
        Calculate a tradability score (0-100) for a pair.
        
        Scoring breakdown:
        - ADX (trend strength): 0-25 points
        - Volatility (ATR%): 0-20 points
        - Volume: 0-15 points
        - RSI positioning: 0-15 points
        - Signal presence: 0-15 points
        - MACD alignment: 0-10 points
        """
        score = 0
        
        # ADX: Higher is better for trending (sweet spot 25-50)
        if adx >= 25:
            adx_score = min(25, (adx - 10) * 0.8)
        else:
            adx_score = adx * 0.5
        score += adx_score
        
        # Volatility: Higher ATR% means more profit potential (sweet spot 2-8%)
        if atr_pct >= 1.5:
            vol_score = min(20, atr_pct * 3)
        else:
            vol_score = atr_pct * 5
        score += vol_score
        
        # Volume: Above average is good
        if volume_ratio >= 1.0:
            vol_mult_score = min(15, volume_ratio * 7)
        else:
            vol_mult_score = volume_ratio * 10
        score += vol_mult_score
        
        # RSI: Best when not at extremes (40-60 is ideal for entries)
        if 35 <= rsi <= 65:
            rsi_score = 15
        elif 25 <= rsi <= 75:
            rsi_score = 10
        else:
            # Extremes can be good for reversal plays
            rsi_score = 8 if (rsi < 25 or rsi > 75) else 5
        score += rsi_score
        
        # Signal presence: Recent signal is highly valuable
        if signal != 0:
            score += 15
        
        # MACD alignment with trend
        if (macd_bullish and trend_strength > 0.5) or (not macd_bullish and trend_strength > 0.5):
            score += 10
        elif macd_bullish or not macd_bullish:
            score += 5
        
        return min(100, max(0, score))

    async def _autonomous_pair_check(self):
        """Check if we should auto-switch to a better pair (autonomous mode)."""
        import time
        current_time = time.time()
        
        # Only check periodically
        if current_time - type(self)._last_auto_switch_check < type(self).AUTO_SWITCH_INTERVAL:
            return
        
        type(self)._last_auto_switch_check = current_time
        
        # Only in autonomous mode and when not in position
        if self.ai_mode != "autonomous" or self.position is not None:
            return
        
        if not type(self).AUTO_SWITCH_ENABLED:
            return
        
        try:
            # Get full market scan
            scan_data = self._get_market_scan_data_full()
            pairs = scan_data.get("pairs", [])
            
            if not pairs:
                return
            
            # Find current pair's score
            current_score = 0
            for p in pairs:
                if p["symbol"] == self.SYMBOL:
                    current_score = p.get("score", 0)
                    break
            
            best_pair = pairs[0]
            best_score = best_pair.get("score", 0)
            
            # Check if best pair is significantly better
            score_diff = best_score - current_score
            
            if best_pair["symbol"] != self.SYMBOL and score_diff >= type(self).AUTO_SWITCH_MIN_SCORE_DIFF:
                # Auto-switch!
                old_symbol = self.SYMBOL
                switch_result = self._switch_trading_symbol(best_pair["symbol"])
                
                if switch_result.get("success"):
                    msg = (
                        f"🤖 **Autonomous Pair Switch**\n\n"
                        f"Switched: {old_symbol} → {best_pair['symbol']}\n\n"
                        f"**Reason:**\n"
                        f"• Old score: {current_score:.1f}\n"
                        f"• New score: {best_score:.1f} (+{score_diff:.1f})\n\n"
                        f"**{best_pair['symbol']} Stats:**\n"
                        f"• RSI: {best_pair.get('rsi', '--')}\n"
                        f"• ADX: {best_pair.get('adx', '--')}\n"
                        f"• Trend: {best_pair.get('trend', '--')}\n"
                        f"• Signal: {best_pair.get('signal_text', 'none')}\n"
                        f"• Volume: {best_pair.get('volume_ratio', 1):.1f}x avg"
                    )
                    
                    logger.info(f"🤖 Auto-switched from {old_symbol} to {best_pair['symbol']} (score +{score_diff:.1f})")
                    
                    # Send Telegram notification
                    if self.telegram.enabled:
                        await self.telegram.send_message(msg)
                    
                    # Clear and refetch data for new pair
                    self.bars_1m = []
                    self.bars_agg = []
                    await self.fetch_initial_data()
                    
        except Exception as e:
            logger.error(f"Autonomous pair check error: {e}")

    def _get_market_scan_data(self) -> Dict[str, Any]:
        """Get multi-pair market data for the scanner (delegates to full scan)."""
        # Use the full scan which has indicators
        return self._get_market_scan_data_full()
    
    def _get_market_scan_data_simple(self) -> Dict[str, Any]:
        """Get basic multi-pair market data (fast, no indicators)."""
        scan_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT", 
                      "ARBUSDT", "SUIUSDT", "APTUSDT", "OPUSDT", "NEARUSDT",
                      "INJUSDT", "TIAUSDT", "SEIUSDT", "WLDUSDT"]
        
        pairs_data = []
        
        try:
            # Use synchronous ccxt for market scanning (the main exchange is async)
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.mexc({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
            
            for symbol in scan_pairs:
                try:
                    # Fetch ticker data using sync exchange
                    ticker = sync_exchange.fetch_ticker(symbol)
                    
                    # Calculate volatility from high/low
                    high = ticker.get('high', 0) or 0
                    low = ticker.get('low', 0) or 0
                    price = ticker.get('last', 0) or ticker.get('close', 0) or 0
                    
                    if price > 0 and high > 0 and low > 0:
                        volatility = (high - low) / price * 100
                    else:
                        volatility = 0
                    
                    change = ticker.get('percentage', 0) or 0
                    volume = ticker.get('quoteVolume', 0) or 0
                    
                    pairs_data.append({
                        "symbol": symbol,
                        "price": price,
                        "change": change,
                        "volatility": volatility,
                        "volume": volume,
                        "high": high,
                        "low": low
                    })
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    continue
            
            # Sort by volatility (highest first)
            pairs_data.sort(key=lambda x: x.get('volatility', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Market scan error: {e}")
        
        return {
            "pairs": pairs_data,
            "current_symbol": self.SYMBOL
        }
    
    def _switch_trading_symbol(self, symbol: str) -> Dict[str, Any]:
        """Switch to a different trading symbol."""
        try:
            # Validate symbol format
            if not symbol.endswith("USDT"):
                symbol = symbol.upper() + "USDT"
            else:
                symbol = symbol.upper()
            
            base = symbol.replace("USDT", "")
            valid_bases = ["BTC", "ETH", "SOL", "LINK", "AVAX", "MATIC", "DOT", "ADA", 
                          "XRP", "DOGE", "ARB", "OP", "APT", "SUI", "NEAR", "INJ", 
                          "TIA", "SEI", "WLD", "PYTH", "JTO", "JUP"]
            
            if base not in valid_bases:
                return {
                    "success": False,
                    "error": f"Invalid symbol. Valid: {', '.join(valid_bases)}"
                }
            
            # Check if we have a position
            if self.position:
                return {
                    "success": False,
                    "error": "Cannot switch while in a position. Close position first."
                }
            
            old_symbol = self.SYMBOL
            self.SYMBOL = symbol
            
            # Clear cached data
            self.latest_candles = {}
            if hasattr(self, 'price_cache') and self.price_cache:
                self.price_cache.clear()
            
            # Save to config for persistence across restarts
            self._save_persisted_symbol()
            
            logger.info(f"🔄 Switched from {old_symbol} to {symbol}")
            
            return {
                "success": True,
                "message": f"Switched from {old_symbol} to {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Symbol switch error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Cache for AI market analysis to prevent flip-flopping recommendations
    _ai_market_cache = {
        "recommendation": None,
        "best_pair": None,
        "timestamp": None,
        "top_pair_at_analysis": None
    }
    AI_MARKET_CACHE_DURATION = 300  # 5 minutes
    
    def _ai_analyze_all_markets(self) -> Dict[str, Any]:
        """Use AI to analyze all market pairs and recommend the best one.
        
        Caches the recommendation for 5 minutes to prevent flip-flopping.
        Only refreshes early if the top volatile pair changes significantly.
        """
        try:
            # Get market scan data
            scan_data = self._get_market_scan_data()
            pairs = scan_data.get('pairs', [])
            
            if not pairs:
                return {"recommendation": "No market data available.", "best_pair": ""}
            
            # Check if we have a valid cached recommendation
            now = time.time()
            cache = self._ai_market_cache
            
            if cache["recommendation"] and cache["timestamp"]:
                cache_age = now - cache["timestamp"]
                
                # Check if cache is still valid (under 5 minutes)
                if cache_age < self.AI_MARKET_CACHE_DURATION:
                    # Check if market conditions changed significantly
                    current_top = pairs[0]["symbol"] if pairs else None
                    cached_top = cache.get("top_pair_at_analysis")
                    
                    # Only invalidate if top pair changed
                    if current_top == cached_top:
                        remaining = int(self.AI_MARKET_CACHE_DURATION - cache_age)
                        logger.debug(f"Using cached AI recommendation (expires in {remaining}s)")
                        return {
                            "recommendation": cache["recommendation"],
                            "best_pair": cache["best_pair"],
                            "cached": True,
                            "cache_expires_in": remaining
                        }
                    else:
                        logger.info(f"Market leader changed: {cached_top} → {current_top}, refreshing AI analysis")
            
            # Build analysis prompt
            pairs = scan_data.get('pairs', [])
            
            if not pairs:
                return {"recommendation": "No market data available.", "best_pair": ""}
            
            # Build analysis prompt with rich indicator data
            pairs_summary = []
            for p in pairs[:10]:  # Top 10 pairs
                signal_txt = ""
                if p.get('signal') == 1:
                    signal_txt = " [LONG SIGNAL!]"
                elif p.get('signal') == -1:
                    signal_txt = " [SHORT SIGNAL!]"
                
                pairs_summary.append(
                    f"- {p['symbol']}: ${p['price']:,.2f} | {p['change']:+.2f}% | "
                    f"Score: {p.get('score', 0):.0f}/100 | RSI: {p.get('rsi', 50):.0f} | "
                    f"ADX: {p.get('adx', 0):.0f} | Trend: {p.get('trend', 'n/a')} | "
                    f"Vol: {p.get('volume_ratio', 1):.1f}x{signal_txt}"
                )
            
            current_balance = self.balance
            risk_pct = self.RISK_PCT * 100
            current_symbol = self.SYMBOL
            
            # Find current symbol's score
            current_score = 0
            for p in pairs:
                if p['symbol'] == current_symbol:
                    current_score = p.get('score', 0)
                    break

            prompt = f"""You are Julaba's AI trading advisor. Analyze these market pairs and recommend the best one for trading.

**Current Trading Setup:**
- Currently trading: {current_symbol} (Score: {current_score:.0f}/100)
- Account Balance: ${current_balance:,.2f}
- Risk per trade: {risk_pct:.1f}%
- Position: {'In position' if self.position else 'No position'}
- Auto-switch enabled: {'Yes' if type(self).AUTO_SWITCH_ENABLED else 'No'}

**Market Pairs (sorted by tradability score):**
{chr(10).join(pairs_summary)}

**Scoring Criteria (0-100):**
- ADX: Trend strength (25+ = strong trend)
- RSI: 40-60 ideal for entries, extremes for reversals  
- Volume: Above 1.0x average is good
- Signal: Recent LONG/SHORT crossover signals

**Your Task:**
1. Recommend the BEST pair to trade NOW
2. Explain WHY with specific indicator values
3. If current pair is optimal, confirm it
4. Consider: score difference must be meaningful (>15 pts) to switch

Keep response under 150 words. Be direct and actionable."""

            recommendation = self.ai_filter._generate_content(prompt)
            
            if recommendation:
                # Try to extract the recommended symbol
                best_pair = ""
                for p in pairs:
                    if p['symbol'].replace('USDT', '') in recommendation.upper():
                        best_pair = p['symbol']
                        break
                
                # Cache the result
                type(self)._ai_market_cache = {
                    "recommendation": recommendation,
                    "best_pair": best_pair,
                    "timestamp": time.time(),
                    "top_pair_at_analysis": pairs[0]['symbol'] if pairs else None
                }
                
                return {
                    "recommendation": recommendation,
                    "best_pair": best_pair,
                    "cached": False,
                    "cache_expires_in": type(self).AI_MARKET_CACHE_DURATION
                }
            else:
                return {
                    "recommendation": "AI analysis temporarily unavailable.",
                    "best_pair": "",
                    "cached": False
                }
                
        except Exception as e:
            logger.error(f"AI market analysis error: {e}")
            return {
                "recommendation": f"Error analyzing markets: {str(e)}",
                "best_pair": ""
            }
    
    def _get_system_params(self) -> Dict[str, Any]:
        """Get all configurable system parameters."""
        return {
            "risk_pct": self.RISK_PCT,
            "atr_mult": self.ATR_MULT,
            "tp1_r": self.TP1_R,
            "tp2_r": self.TP2_R,
            "tp3_r": self.TP3_R,
            "tp1_pct": self.TP1_PCT,
            "tp2_pct": self.TP2_PCT,
            "tp3_pct": self.TP3_PCT,
            "trail_trigger_r": self.TRAIL_TRIGGER_R,
            "trail_offset_r": self.TRAIL_OFFSET_R,
            "ai_mode": self.ai_mode,
            "ai_confidence": self.ai_filter.confidence_threshold,
            "ai_scan_interval": self.ai_scan_interval,
            "ai_scan_notify_opportunities_only": self.ai_scan_notify_opportunities_only,
            "ai_scan_quiet_interval": self.ai_scan_quiet_interval,
            "summary_interval": self.summary_interval,
            "symbol": self.SYMBOL,
            "paused": self.paused,
            "daily_loss_limit": self.daily_loss_limit,
            "daily_loss_triggered": self.daily_loss_triggered,
            "dry_run_mode": self.dry_run_mode,
            "auto_switch": type(self).AUTO_SWITCH_ENABLED,
            "auto_switch_interval": type(self).AUTO_SWITCH_INTERVAL,
            "auto_switch_min_diff": type(self).AUTO_SWITCH_MIN_SCORE_DIFF,
        }
    
    def _set_system_param(self, param: str, value: Any) -> Dict[str, Any]:
        """Set a system parameter. Returns success status and message."""
        param = param.lower().replace(" ", "_")
        
        try:
            if param == "risk_pct":
                val = float(value)
                if 0.001 <= val <= 0.1:  # 0.1% to 10%
                    self.RISK_PCT = val
                    return {"success": True, "message": f"Risk set to {val*100:.1f}%"}
                return {"success": False, "message": "Risk must be between 0.1% and 10%"}
            
            elif param == "atr_mult":
                val = float(value)
                if 0.5 <= val <= 5.0:
                    self.ATR_MULT = val
                    return {"success": True, "message": f"ATR multiplier set to {val}"}
                return {"success": False, "message": "ATR mult must be between 0.5 and 5.0"}
            
            elif param == "ai_confidence":
                val = float(value)
                if 0.1 <= val <= 1.0:
                    self.ai_filter.confidence_threshold = val
                    return {"success": True, "message": f"AI confidence threshold set to {val*100:.0f}%"}
                return {"success": False, "message": "Confidence must be between 10% and 100%"}
            
            elif param == "ai_mode":
                if value in ["filter", "advisory", "autonomous", "hybrid"]:
                    self.ai_mode = value
                    return {"success": True, "message": f"AI mode set to {value}"}
                return {"success": False, "message": "Mode must be: filter, advisory, autonomous, or hybrid"}
            
            elif param in ["tp1_r", "tp2_r", "tp3_r"]:
                val = float(value)
                if 0.5 <= val <= 10.0:
                    setattr(self, param.upper(), val)
                    return {"success": True, "message": f"{param.upper()} set to {val}R"}
                return {"success": False, "message": "TP must be between 0.5R and 10R"}
            
            elif param == "ai_scan_interval":
                val = int(value)
                if 30 <= val <= 3600:
                    self.ai_scan_interval = val
                    return {"success": True, "message": f"AI scan interval set to {val}s"}
                return {"success": False, "message": "Interval must be 30-3600 seconds"}
            
            elif param == "ai_scan_notify_opportunities_only":
                self.ai_scan_notify_opportunities_only = str(value).lower() in ["true", "1", "yes"]
                return {"success": True, "message": f"Scan notifications: {'opportunities only' if self.ai_scan_notify_opportunities_only else 'all scans'}"}
            
            elif param == "ai_scan_quiet_interval":
                val = int(value)
                if 300 <= val <= 7200:
                    self.ai_scan_quiet_interval = val
                    return {"success": True, "message": f"Quiet notification interval set to {val}s ({val//60} min)"}
                return {"success": False, "message": "Quiet interval must be 300-7200 seconds (5-120 min)"}
            
            elif param == "paused":
                self.paused = str(value).lower() in ["true", "1", "yes"]
                return {"success": True, "message": f"Bot {'paused' if self.paused else 'resumed'}"}
            
            elif param == "daily_loss_limit":
                val = float(value)
                if 0.01 <= val <= 0.20:  # 1% to 20%
                    self.daily_loss_limit = val
                    return {"success": True, "message": f"Daily loss limit set to {val*100:.1f}%"}
                return {"success": False, "message": "Daily loss limit must be between 1% and 20%"}
            
            elif param == "dry_run" or param == "dry_run_mode":
                self.dry_run_mode = str(value).lower() in ["true", "1", "yes"]
                return {"success": True, "message": f"Dry-run mode {'enabled' if self.dry_run_mode else 'disabled'}"}
            
            elif param == "reset_daily_loss":
                self.daily_loss_triggered = False
                self.stats.today_pnl = 0.0
                return {"success": True, "message": "Daily loss circuit breaker reset"}
            
            elif param == "symbol":
                # Validate symbol format (should be like "BTC/USDT", "ETH/USDT", etc.)
                new_symbol = str(value).upper().strip()
                if "/" not in new_symbol:
                    # Try to add /USDT if not present
                    new_symbol = f"{new_symbol}/USDT"
                
                # Common valid symbols
                valid_bases = ["BTC", "ETH", "SOL", "LINK", "AVAX", "MATIC", "DOT", "ADA", "XRP", "DOGE", "SHIB", "ARB", "OP", "APT", "SUI", "PEPE", "WIF", "BONK", "INJ", "TIA", "SEI", "NEAR", "FTM", "ATOM", "UNI", "AAVE", "LTC", "BCH", "ETC", "FIL", "RENDER", "TAO", "WLD", "PYTH", "JTO", "JUP"]
                base = new_symbol.split("/")[0]
                
                if base not in valid_bases:
                    return {"success": False, "message": f"Unknown symbol {new_symbol}. Supported: BTC, ETH, SOL, LINK, AVAX, MATIC, DOT, ADA, XRP, ARB, OP, APT, SUI, NEAR, UNI, AAVE, and more"}
                
                if self.position is not None:
                    return {"success": False, "message": f"Cannot change symbol while in a position. Close position first."}
                
                old_symbol = self.SYMBOL
                self.SYMBOL = new_symbol
                self.ai_filter.default_symbol = new_symbol
                
                # Reset cached data for new symbol
                self.bars_1m = []
                self.bars_agg = pd.DataFrame()
                self.cached_last_price = 0
                self.cached_last_atr = 0
                
                # Save to config for persistence across restarts
                self._save_persisted_symbol()
                
                logger.info(f"🔄 Symbol changed from {old_symbol} to {new_symbol}")
                return {"success": True, "message": f"Symbol changed to {new_symbol}. Fetching new data..."}
            
            elif param == "auto_switch":
                type(self).AUTO_SWITCH_ENABLED = str(value).lower() in ["true", "1", "yes", "on"]
                status = "enabled" if type(self).AUTO_SWITCH_ENABLED else "disabled"
                return {"success": True, "message": f"🔄 Autonomous pair switching {status}"}
            
            elif param == "auto_switch_interval":
                val = int(value)
                if 60 <= val <= 3600:
                    type(self).AUTO_SWITCH_INTERVAL = val
                    return {"success": True, "message": f"Auto-switch check interval set to {val}s ({val//60} min)"}
                return {"success": False, "message": "Interval must be 60-3600 seconds"}
            
            elif param == "auto_switch_min_diff":
                val = int(value)
                if 5 <= val <= 50:
                    type(self).AUTO_SWITCH_MIN_SCORE_DIFF = val
                    return {"success": True, "message": f"Auto-switch minimum score difference set to {val} points"}
                return {"success": False, "message": "Min difference must be 5-50 points"}
            
            else:
                return {"success": False, "message": f"Unknown parameter: {param}"}
                
        except (ValueError, TypeError) as e:
            return {"success": False, "message": f"Invalid value: {e}"}
    
    def _get_full_system_state(self) -> Dict[str, Any]:
        """Get complete system state for AI context - SINGLE SOURCE OF TRUTH.
        
        All data flows through this method to ensure consistency between
        what the AI sees and what Telegram commands display.
        """
        ml_stats = self._get_ml_stats()  # Already normalized
        regime_info = self._get_regime()
        
        # Get market scan data for AI context (top 5 pairs by score)
        market_scan = {}
        try:
            scan_data = self._get_market_scan_data()
            if scan_data and scan_data.get('pairs'):
                pairs = scan_data['pairs'][:5]  # Top 5 by score
                market_scan = {
                    "current_symbol": scan_data.get('current_symbol', ''),
                    "auto_switch_enabled": type(self).AUTO_SWITCH_ENABLED,
                    "best_pair": scan_data.get('best_pair', {}).get('symbol') if scan_data.get('best_pair') else None,
                    "top_pairs": [
                        {
                            "symbol": p['symbol'],
                            "price": p['price'],
                            "change": p['change'],
                            "score": p.get('score', 0),
                            "rsi": p.get('rsi', 50),
                            "adx": p.get('adx', 0),
                            "trend": p.get('trend', 'n/a'),
                            "signal": p.get('signal_text', 'none')
                        }
                        for p in pairs
                    ]
                }
        except Exception as e:
            logger.debug(f"Market scan for context: {e}")
        
        return {
            "parameters": self._get_system_params(),
            "status": self._get_status(),
            "position": self._get_positions(),
            "pnl": self._get_pnl(),
            "market": self._get_market(),
            "market_scan": market_scan,
            "ml": ml_stats,  # Use normalized stats directly
            "regime": regime_info.get("regime", "unknown") if regime_info else "unknown",
            "regime_details": regime_info if regime_info else {},
            "signals": self._get_signals()[-3:] if self._get_signals() else [],
            "trades": self._get_trades()[-5:] if self._get_trades() else [],
            "intelligence": self._get_intelligence(),
        }
    
    def _toggle_summary_notifications(self) -> bool:
        """Toggle summary notifications on/off. Returns new state."""
        self.summary_notifications_enabled = not self.summary_notifications_enabled
        return self.summary_notifications_enabled
    
    async def _chat_with_ai(self, message: str, context: str) -> str:
        """Chat with AI through Telegram."""
        return await self.ai_filter.chat(message, context)
    
    async def _execute_ai_trade(self, side: str) -> Dict[str, Any]:
        """Execute a trade requested by AI chat.
        
        Args:
            side: 'long' or 'short'
            
        Returns:
            Dict with success status and message
        """
        try:
            # Check if already have a position
            if self.position:
                return {
                    "success": False,
                    "message": f"Already have an open {self.position.side.upper()} position. Close it first."
                }
            
            # Check if paused
            if self.paused:
                return {
                    "success": False,
                    "message": "Bot is paused. Use /resume to enable trading."
                }
            
            # Get current price and ATR
            price = self._last_price
            atr = self._calculate_atr()
            
            if not price or atr <= 0:
                return {
                    "success": False,
                    "message": "Cannot execute - no price data available yet."
                }
            
            # Execute the trade
            signal = 1 if side.lower() == "long" else -1
            await self._open_position(signal, price, atr, source="ai_chat")
            
            return {
                "success": True,
                "message": f"Opened {side.upper()} position at ${price:.4f}",
                "price": price,
                "side": side.upper()
            }
            
        except Exception as e:
            logger.error(f"AI trade execution error: {e}")
            return {
                "success": False,
                "message": f"Trade failed: {str(e)}"
            }
    
    async def _close_ai_trade(self) -> Dict[str, Any]:
        """Close position requested by AI chat.
        
        Returns:
            Dict with success status and message
        """
        try:
            if not self.position:
                return {
                    "success": False,
                    "message": "No open position to close."
                }
            
            price = self._last_price
            if not price:
                return {
                    "success": False,
                    "message": "Cannot close - no price data available."
                }
            
            side = self.position.side.upper()
            await self._close_position("AI Chat Request", price)
            
            return {
                "success": True,
                "message": f"Closed {side} position at ${price:.4f}"
            }
            
        except Exception as e:
            logger.error(f"AI close error: {e}")
            return {
                "success": False,
                "message": f"Close failed: {str(e)}"
            }
    
    def _get_status(self) -> Dict[str, Any]:
        """Get bot status for Telegram."""
        if self.start_time:
            uptime = datetime.now(timezone.utc) - self.start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        else:
            uptime_str = "Starting..."
        
        # Get current price and ATR
        current_price = self.cached_last_price or 0
        # Fall back to bars data if no cached price yet
        if current_price == 0 and hasattr(self, 'bars_agg') and len(self.bars_agg) > 0:
            current_price = float(self.bars_agg.iloc[-1]['close'])
        current_atr = self.cached_last_atr or 0
        
        # Position info
        has_position = self.position is not None
        position_side = self.position.side.upper() if has_position else "None"
        position_pnl = self.position.unrealized_pnl(current_price) if has_position and current_price else 0
        
        # Stats - use correct attribute names from TradeStats
        total_trades = self.stats.total_trades
        wins = self.stats.winning_trades
        losses = self.stats.losing_trades
        win_rate = self.stats.win_rate * 100  # Convert to percentage
        
        return {
            "connected": self.exchange is not None,
            "symbol": self.SYMBOL,
            "uptime": uptime_str,
            "mode": "Paper" if self.paper_mode else "Live",
            "paused": self.paused,
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "current_price": current_price,
            "atr": current_atr,
            "has_position": has_position,
            "position_side": position_side,
            "position_pnl": position_pnl,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": self.stats.total_pnl,
            "signals_checked": len(self.signal_history)
        }
    
    def _get_ml_status(self) -> Dict[str, Any]:
        """Get ML model status for dashboard."""
        if not self.ml_predictor or not self.ml_predictor.is_loaded:
            return {
                "loaded": False,
                "status": "Not Available",
                "accuracy": 0,
                "samples": 0,
                "influence": 0,
                "last_prediction": None
            }
        
        # Use metrics attribute instead of model_metadata
        metrics = getattr(self.ml_predictor, 'metrics', {})
        last_pred = None
        if hasattr(self.ml_predictor, 'prediction_log') and self.ml_predictor.prediction_log:
            last_pred = self.ml_predictor.prediction_log[-1] if self.ml_predictor.prediction_log else None
        
        return {
            "loaded": True,
            "status": "Active (Advisory)" if metrics.get('accuracy', 0) > 0 else "Loaded",
            "accuracy": metrics.get('accuracy', 0),
            "samples": metrics.get('total_samples', 0),
            "influence": 0.0,  # Currently advisory only
            "model_path": str(self.ml_predictor.model_path),
            "features": len(self.ml_predictor.feature_columns),
            "last_prediction": last_pred
        }

    def _get_system_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent system logs for dashboard."""
        logs = []
        log_file = Path(__file__).parent / "julaba.log"
        
        try:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # Get last N lines
                    recent_lines = lines[-count:] if len(lines) > count else lines
                    
                    for line in recent_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse log line: "2026-01-09 20:42:57 [INFO] Julaba: message"
                        try:
                            parts = line.split(' ', 3)
                            if len(parts) >= 4:
                                timestamp = f"{parts[0]} {parts[1]}"
                                level = parts[2].strip('[]')
                                message = parts[3] if len(parts) > 3 else ""
                                
                                logs.append({
                                    "time": timestamp,
                                    "level": level,
                                    "message": message[:200]  # Truncate long messages
                                })
                            else:
                                logs.append({
                                    "time": "",
                                    "level": "INFO",
                                    "message": line[:200]
                                })
                        except Exception:
                            logs.append({
                                "time": "",
                                "level": "INFO", 
                                "message": line[:200]
                            })
        except Exception as e:
            logs.append({
                "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "level": "ERROR",
                "message": f"Failed to read logs: {e}"
            })
        
        return logs

    def _get_positions(self) -> List[Dict[str, Any]]:
        """Get positions for Telegram."""
        if not self.position:
            return []
        
        return [{
            "symbol": self.position.symbol,
            "side": self.position.side.upper(),
            "entry": self.position.entry_price,
            "size": self.position.remaining_size,
            "pnl": self.position.unrealized_pnl(self._last_price or self.position.entry_price)
        }]
    
    def _get_current_position_dict(self) -> Optional[Dict[str, Any]]:
        """Get current position as dict for dashboard."""
        if not self.position:
            return None
        return {
            "symbol": self.position.symbol,
            "side": self.position.side.upper(),
            "entry": self.position.entry_price,
            "size": self.position.remaining_size,
            "pnl": self.position.unrealized_pnl(self._last_price or self.position.entry_price),
            "stop_loss": self.position.stop_loss,
            "take_profit": self.position.take_profit
        }
    
    def _get_pnl(self) -> Dict[str, Any]:
        """Get P&L for Telegram."""
        # Reset daily P&L if new day
        now = datetime.utcnow()
        if now.date() > self.stats.last_reset.date():
            self.stats.today_pnl = 0.0
            self.stats.last_reset = now
        
        return {
            "today": self.stats.today_pnl,
            "total": self.stats.total_pnl,
            "win_rate": self.stats.win_rate,
            "trades": self.stats.total_trades,
            "winning": self.stats.winning_trades,
            "max_win": self.stats.max_win,
            "max_loss": self.stats.max_loss,
            "avg_trade": self.stats.total_pnl / max(1, self.stats.total_trades)
        }
    
    def _get_balance(self) -> Dict[str, Any]:
        """Get balance info for Telegram."""
        change = self.balance - self.initial_balance
        change_pct = (change / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        return {
            "current": self.balance,
            "initial": self.initial_balance,
            "change": change,
            "change_pct": change_pct
        }
    
    def _get_trades(self) -> List[Dict]:
        """Get trade history for Telegram."""
        return self.trade_history
    
    def _get_market(self) -> Dict[str, Any]:
        """Get market info for Telegram."""
        return {
            "symbol": self.SYMBOL,
            "price": self.cached_last_price,
            "change_24h": self.cached_last_ticker.get('percentage', 0),
            "volume_24h": self.cached_last_ticker.get('quoteVolume', 0),
            "high_24h": self.cached_last_ticker.get('high', 0),
            "low_24h": self.cached_last_ticker.get('low', 0),
            "atr": self.cached_last_atr
        }
    
    def _get_signals(self) -> List[Dict]:
        """Get signal history for Telegram."""
        return self.signal_history
    
    async def _do_stop(self):
        """Stop the bot from Telegram command."""
        self.running = False
    
    def _do_pause(self):
        """Pause trading from Telegram command."""
        self.paused = True
        logger.info("Trading paused via Telegram")
    
    def _do_resume(self):
        """Resume trading from Telegram command."""
        self.paused = False
        logger.info("Trading resumed via Telegram")

    def _get_intelligence(self) -> Dict[str, Any]:
        """Get intelligence summary for Telegram /intel command."""
        result = {
            'drawdown_mode': 'NORMAL',
            'drawdown_pct': 0.0,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'pattern': None,
            'regime': 'UNKNOWN',
            'tradeable': False,
            'ml_status': 'Not trained'
        }
        
        # Drawdown calculation
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
            result['drawdown_pct'] = round(drawdown, 2)
            if drawdown >= 20:
                result['drawdown_mode'] = 'EMERGENCY'
            elif drawdown >= 10:
                result['drawdown_mode'] = 'CAUTIOUS'
            elif drawdown >= 5:
                result['drawdown_mode'] = 'REDUCED'
        
        # Market regime
        if len(self.bars_agg) >= 50:
            regime = get_regime_analysis(self.bars_agg)
            result['regime'] = regime.get('regime', 'UNKNOWN')
            result['tradeable'] = regime.get('tradeable', False)
            result['adx'] = regime.get('adx', 0)
            result['hurst'] = regime.get('hurst', 0.5)
            
            # Pattern detection
            pattern = detect_candlestick_patterns(self.bars_agg)
            if pattern.get('pattern'):
                result['pattern'] = pattern
        
        # ML status
        ml_stats = get_ml_classifier().get_stats()
        result['ml_samples'] = ml_stats.get('total_samples', 0)
        result['ml_trained'] = ml_stats.get('is_trained', False)
        if result['ml_trained']:
            result['ml_status'] = f"Trained ({ml_stats.get('total_samples', 0)} samples)"
        else:
            needed = ml_stats.get('samples_until_training', 50)
            result['ml_status'] = f"Learning ({needed} more needed)"
        
        return result
    
    def _get_ml_stats(self) -> Dict[str, Any]:
        """Get ML classifier stats - SINGLE SOURCE OF TRUTH for both Telegram and AI context."""
        raw = get_ml_classifier().get_stats()
        # Normalize to consistent key names used everywhere
        return {
            'total_samples': raw.get('total_samples', 0),
            'samples': raw.get('total_samples', 0),  # Alias for convenience
            'is_trained': raw.get('is_trained', False),
            'trained': raw.get('is_trained', False),  # Alias for convenience
            'samples_until_training': raw.get('samples_until_training', 50),
            'historical_win_rate': raw.get('historical_win_rate', 0),
            'wins': raw.get('wins', 0),
            'losses': raw.get('losses', 0),
            'top_features': raw.get('top_features', []),
            'model_version': raw.get('model_version', 'v2'),
            'num_features': raw.get('num_features', 22),
        }
    
    def _get_regime(self) -> Dict[str, Any]:
        """Get current market regime for Telegram /regime command."""
        result = {
            'regime': 'UNKNOWN',
            'adx': 0,
            'hurst': 0.5,
            'volatility': 'normal',
            'volatility_ratio': 1.0,
            'tradeable': False,
            'ml_prediction': None,
            'ml_confidence': 0,
            'description': 'Insufficient data'
        }
        
        if len(self.bars_agg) < 50:
            result['description'] = f'Need more data ({len(self.bars_agg)}/50 bars)'
            return result
        
        # Get regime analysis
        regime = get_regime_analysis(self.bars_agg)
        result['regime'] = regime.get('regime', 'UNKNOWN')
        result['adx'] = round(regime.get('adx', 0), 1)
        result['hurst'] = round(regime.get('hurst', 0.5), 3)
        result['tradeable'] = regime.get('tradeable', False)
        
        # Volatility
        from indicator import calculate_volatility_regime
        vol = calculate_volatility_regime(self.bars_agg['close'])
        result['volatility'] = vol.get('regime', 'normal')
        result['volatility_ratio'] = round(vol.get('volatility_ratio', 1.0), 2)
        
        # ML prediction if trained
        ml = get_ml_classifier()
        if ml.is_trained:
            from indicator import compute_ml_features
            features = compute_ml_features(self.bars_agg)
            pred = ml.predict(features)
            if pred:
                result['ml_prediction'] = pred.get('regime')
                result['ml_confidence'] = round(pred.get('confidence', 0) * 100, 1)
        
        # Description
        regime_desc = {
            'STRONG_TRENDING': 'Strong directional move - trend following works well',
            'TRENDING': 'Clear trend - good for momentum strategies',
            'WEAK_TRENDING': 'Weak trend - caution advised',
            'RANGING': 'Sideways market - mean reversion may work',
            'CHOPPY': 'Choppy/noisy - avoid trading'
        }
        result['description'] = regime_desc.get(result['regime'], 'Unknown market condition')
        
        return result

    def _get_indicators_for_dashboard(self) -> Dict[str, Any]:
        """Get current technical indicator values for dashboard."""
        result = {
            'rsi': None,
            'macd_signal': '--',
            'adx': None,
            'atr': None,
            'bb_position': '--',
            'volume_ratio': None
        }
        
        if len(self.bars_agg) < 20:
            return result
        
        try:
            from indicator import calculate_rsi, calculate_atr, calculate_adx
            
            close = self.bars_agg['close']
            
            # RSI
            rsi = calculate_rsi(close, 14)
            if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
                result['rsi'] = float(rsi.iloc[-1])
            
            # ATR
            atr = calculate_atr(self.bars_agg, 14)
            if len(atr) > 0 and not pd.isna(atr.iloc[-1]):
                result['atr'] = float(atr.iloc[-1])
            
            # ADX
            result['adx'] = calculate_adx(self.bars_agg, 14)
            
            # MACD Signal
            if len(self.bars_agg) >= 26:
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                if len(macd) > 0:
                    if macd.iloc[-1] > signal.iloc[-1]:
                        result['macd_signal'] = 'BULLISH' if macd.iloc[-1] > 0 else 'WEAK BULL'
                    else:
                        result['macd_signal'] = 'BEARISH' if macd.iloc[-1] < 0 else 'WEAK BEAR'
            
            # Bollinger Bands position
            if len(close) >= 20:
                sma20 = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                upper = sma20 + 2 * std20
                lower = sma20 - 2 * std20
                current = close.iloc[-1]
                if current >= upper.iloc[-1]:
                    result['bb_position'] = 'ABOVE (OB)'
                elif current <= lower.iloc[-1]:
                    result['bb_position'] = 'BELOW (OS)'
                else:
                    pct = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
                    result['bb_position'] = f'{pct:.0f}%'
            
            # Volume ratio
            if 'volume' in self.bars_agg.columns and len(self.bars_agg) >= 20:
                vol = self.bars_agg['volume']
                avg_vol = vol.rolling(20).mean().iloc[-1]
                if avg_vol > 0:
                    result['volume_ratio'] = float(vol.iloc[-1] / avg_vol)
        except Exception as e:
            logger.debug(f"Error getting indicators for dashboard: {e}")
        
        return result

    def _get_current_signal(self) -> Dict[str, Any]:
        """Get current signal state for dashboard."""
        result = {
            'direction': None,
            'confidence': 0,
            'entry': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # If we have recent signal history, return the latest
        if self.signal_history:
            latest = self.signal_history[-1]
            result['direction'] = latest.get('direction')
            result['confidence'] = latest.get('confidence', 0)
            result['entry'] = latest.get('entry')
            result['stop_loss'] = latest.get('sl')
            result['take_profit'] = latest.get('tp')
        
        return result

    def _get_ohlc_for_chart(self, timeframe: str = '1m') -> List[Dict]:
        """Get OHLC data for live price chart on dashboard."""
        result = []
        
        try:
            # Handle 1m bars (stored as list of dicts)
            if timeframe == '1m':
                if hasattr(self, 'bars_1m') and len(self.bars_1m) > 0:
                    # bars_1m is a list of dicts
                    data = list(self.bars_1m)[-100:]
                    for bar in data:
                        ts = bar.get('timestamp', 0)
                        # Ensure timestamp is in milliseconds
                        if isinstance(ts, (int, float)) and ts < 10000000000:
                            ts = int(ts * 1000)
                        candle = {
                            't': int(ts),
                            'o': float(bar.get('open', 0)),
                            'h': float(bar.get('high', 0)),
                            'l': float(bar.get('low', 0)),
                            'c': float(bar.get('close', 0)),
                            'v': float(bar.get('volume', 0))
                        }
                        result.append(candle)
                    return result
            
            # Handle DataFrames (3m, 15m, 1h)
            if timeframe == '3m':
                df = self.bars_agg
            elif timeframe == '15m':
                df = self.bars_15m if hasattr(self, 'bars_15m') and isinstance(self.bars_15m, pd.DataFrame) and len(self.bars_15m) > 0 else None
            elif timeframe == '5m':
                # Resample 1m to 5m
                if hasattr(self, 'bars_1m') and len(self.bars_1m) >= 5:
                    df_1m = pd.DataFrame(list(self.bars_1m))
                    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
                    df_1m.set_index('timestamp', inplace=True)
                    df = df_1m.resample('5min', label='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna().reset_index()
                else:
                    df = None
            else:
                df = self.bars_agg
            
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return result
            
            # Get last 100 candles
            df_slice = df.tail(100)
            
            for _, row in df_slice.iterrows():
                # Get timestamp from 'timestamp' column
                ts = row.get('timestamp', row.get('Timestamp', None))
                if ts is None:
                    continue
                
                # Convert to milliseconds
                if hasattr(ts, 'timestamp'):
                    ts = int(ts.timestamp() * 1000)
                elif isinstance(ts, (int, float)):
                    if ts < 10000000000:  # seconds, not ms
                        ts = int(ts * 1000)
                    else:
                        ts = int(ts)
                else:
                    ts = int(pd.Timestamp(ts).timestamp() * 1000)
                
                candle = {
                    't': ts,
                    'o': float(row.get('open', row.get('Open', 0))),
                    'h': float(row.get('high', row.get('High', 0))),
                    'l': float(row.get('low', row.get('Low', 0))),
                    'c': float(row.get('close', row.get('Close', 0))),
                    'v': float(row.get('volume', row.get('Volume', 0)))
                }
                result.append(candle)
        except Exception as e:
            logger.debug(f"Error getting OHLC for chart: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return result

    def _set_ai_mode(self, mode: str) -> bool:
        """Set AI trading mode."""
        valid_modes = ["filter", "advisory", "autonomous", "hybrid"]
        if mode.lower() in valid_modes:
            self.ai_mode = mode.lower()
            logger.info(f"AI mode changed to: {self.ai_mode}")
            return True
        return False
    
    async def _confirm_ai_trade(self):
        """Confirm pending AI trade (for advisory/hybrid mode)."""
        if self.pending_ai_trade and self.position is None:
            trade = self.pending_ai_trade
            self.pending_ai_trade = None
            
            logger.info(f"AI trade CONFIRMED by user: {trade['action']}")
            await self._open_position(
                trade['signal'],
                self.cached_last_price,
                self.cached_last_atr,
                source="ai_confirmed"
            )
            return True
        return False
    
    async def _reject_ai_trade(self):
        """Reject pending AI trade."""
        if self.pending_ai_trade:
            logger.info(f"AI trade REJECTED by user: {self.pending_ai_trade['action']}")
            self.pending_ai_trade = None
            return True
        return False

    # ============== NEW ENHANCEMENT METHODS ==============
    
    def _get_risk_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics for Telegram."""
        stats = self.risk_manager.get_stats()
        can_trade = self.risk_manager.can_trade(self.balance, self.initial_balance)
        risk_info = self.risk_manager.get_adjusted_risk(
            self.balance, 
            self.peak_balance,
            self.cached_last_atr / self.cached_last_price * 100 if self.cached_last_price > 0 else 1.0
        )
        return {
            **stats,
            'can_trade': can_trade['allowed'],
            'can_trade_reason': can_trade['reason'],
            'adjusted_risk': risk_info['adjusted_risk'],
            'dd_mode': risk_info['dd_mode'],
            'kelly_risk': risk_info['kelly_risk']
        }
    
    def _get_mtf_analysis(self) -> Dict[str, Any]:
        """Get multi-timeframe analysis for Telegram."""
        if len(self.bars_agg) < 20:
            return {'error': 'Insufficient data for MTF analysis'}
        
        # Get last signal
        try:
            signals_df = generate_signals(self.bars_agg)
            last_signal = int(signals_df.iloc[-1].get("Side", 0))
        except:
            last_signal = 0
        
        # Run MTF analysis
        result = self.mtf_analyzer.analyze(self.bars_agg, proposed_signal=last_signal)
        return result
    
    async def _run_backtest(self, days: int = 7) -> Dict[str, Any]:
        """Run backtest on recent historical data."""
        from backtest import BacktestEngine, fetch_historical_data
        
        try:
            if not self.exchange:
                await self.connect()
            
            # Fetch historical data
            df = await fetch_historical_data(
                self.exchange,
                self.SYMBOL,
                timeframe='3m',
                days=days
            )
            
            if len(df) < 100:
                return {'error': f'Insufficient data: only {len(df)} bars'}
            
            # Run backtest
            engine = BacktestEngine(
                initial_balance=10000,
                risk_pct=self.RISK_PCT,
                atr_mult=self.ATR_MULT,
                tp1_r=self.TP1_R,
                tp2_r=self.TP2_R,
                tp3_r=self.TP3_R
            )
            
            result = engine.run(df)
            
            # Run Monte Carlo
            mc_results = engine.monte_carlo(result, simulations=500)
            
            return {
                'success': True,
                'days': days,
                'bars': len(df),
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'total_pnl_pct': result.total_pnl_pct,
                'max_drawdown_pct': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'monte_carlo': mc_results,
                'formatted': engine.format_results(result)
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    def _get_chart(self) -> Optional[bytes]:
        """Generate a chart of recent price action."""
        if len(self.bars_agg) < 20:
            return None
        
        entry_price = None
        entry_side = None
        stop_loss = None
        take_profits = None
        
        if self.position:
            entry_price = self.position.entry_price
            entry_side = self.position.side
            stop_loss = self.position.stop_loss
            take_profits = [self.position.tp1, self.position.tp2, self.position.tp3]
        
        return self.chart_generator.generate_candlestick_chart(
            self.bars_agg,
            symbol=self.SYMBOL,
            entry_price=entry_price,
            entry_side=entry_side,
            stop_loss=stop_loss,
            take_profits=take_profits
        )
    
    async def _fetch_higher_timeframes(self):
        """Fetch 15m and 1H data for MTF analysis."""
        if not self.exchange:
            return
        
        now = datetime.now(timezone.utc)
        
        # Only update every 15 minutes
        if self.last_htf_update and (now - self.last_htf_update).total_seconds() < 900:
            return
        
        try:
            # Fetch 15m bars
            ohlcv_15m = await self.exchange.fetch_ohlcv(self.SYMBOL, '15m', limit=100)
            if ohlcv_15m:
                self.bars_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.mtf_analyzer.update_data(df_15m=self.bars_15m)
            
            # Fetch 1H bars
            ohlcv_1h = await self.exchange.fetch_ohlcv(self.SYMBOL, '1h', limit=50)
            if ohlcv_1h:
                self.bars_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.mtf_analyzer.update_data(df_1h=self.bars_1h)
            
            self.last_htf_update = now
            logger.debug(f"Updated HTF data: 15m={len(self.bars_15m)} bars, 1H={len(self.bars_1h)} bars")
            
        except Exception as e:
            logger.warning(f"Failed to fetch HTF data: {e}")
    
    def _update_equity_curve(self):
        """Update equity curve after balance changes."""
        self.equity_curve.append(self.balance)
        # Keep last 1000 points
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]

    async def connect(self):
        """Connect to the exchange."""
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        
        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        }
        
        if api_key and api_secret and not self.paper_mode:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            logger.info("Using authenticated connection")
        else:
            logger.info("Using public connection (paper trading)")
        
        self.exchange = ccxt.mexc(config)
        
        # Load markets
        await self.exchange.load_markets()
        logger.info(f"Connected to MEXC | Symbol: {self.SYMBOL}")
    
    async def fetch_initial_data(self):
        """Fetch initial historical data for warmup."""
        logger.info(f"Fetching initial data ({self.WARMUP_BARS} bars)...")
        
        # Need WARMUP_BARS * 3 (for 3:1 aggregation) + extra buffer for aggregation
        bars_to_fetch = (self.WARMUP_BARS * 3) + 10
        ohlcv = await self.exchange.fetch_ohlcv(
            self.SYMBOL,
            self.BASE_TF,
            limit=bars_to_fetch
        )
        
        for candle in ohlcv:
            self.bars_1m.append({
                "timestamp": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        
        self._aggregate_bars()
        logger.info(f"Loaded {len(self.bars_agg)} aggregated bars")
    
    async def pretrain_ml_model(self):
        """Pre-train ML model on historical data using simulated trades."""
        logger.info("🧠 Starting ML pre-training on historical data...")
        
        # Connect if not already connected
        if not self.exchange:
            await self.connect()
        
        # Fetch extended historical data (500 bars = ~25 hours of 3m data)
        logger.info("Fetching extended historical data for ML pre-training...")
        ohlcv = await self.exchange.fetch_ohlcv(
            self.SYMBOL,
            "3m",  # Direct 3m bars for more history
            limit=500
        )
        
        if len(ohlcv) < 100:
            logger.warning("Insufficient historical data for pre-training")
            return
        
        # Build DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Simulate trades on historical data
        ml_classifier = get_ml_classifier()
        samples_before = len(ml_classifier.training_data)
        
        # Use a sliding window to generate training samples
        window_size = 100  # Need at least 100 bars for feature extraction
        
        for i in range(window_size, len(df) - 20):  # Leave 20 bars for outcome
            window_df = df.iloc[i-window_size:i].copy()
            
            # Generate signal for this window
            signals_df = generate_signals(window_df)
            signal = int(signals_df.iloc[-1].get("Side", 0))
            
            if signal == 0:
                continue
            
            # Simulate trade outcome: check if price moved in favorable direction
            entry_price = df.iloc[i]['close']
            future_prices = df.iloc[i:i+20]['close'].values
            
            if signal == 1:  # Long
                max_gain = (max(future_prices) - entry_price) / entry_price * 100
                max_loss = (entry_price - min(future_prices)) / entry_price * 100
                # Win if max gain > 0.5% and > max loss
                won = max_gain > 0.5 and max_gain > max_loss
            else:  # Short
                max_gain = (entry_price - min(future_prices)) / entry_price * 100
                max_loss = (max(future_prices) - entry_price) / entry_price * 100
                won = max_gain > 0.5 and max_gain > max_loss
            
            # Record sample
            ml_classifier.record_sample(window_df, won)
        
        samples_after = len(ml_classifier.training_data)
        new_samples = samples_after - samples_before
        
        logger.info(f"🧠 ML Pre-training complete: {new_samples} new samples added")
        logger.info(f"🧠 Total samples: {samples_after}, Trained: {ml_classifier.is_trained}")
        
        if self.telegram and self.telegram.enabled:
            await self.telegram.send_message(
                f"🧠 *ML Pre-Training Complete*\\n\\n"
                f"New samples: `{new_samples}`\\n"
                f"Total samples: `{samples_after}`\\n"
                f"Model trained: `{'Yes' if ml_classifier.is_trained else 'No'}`"
            )
    
    def _aggregate_bars(self):
        """Aggregate 1m bars to 3m bars."""
        if len(self.bars_1m) < self.AGG_TF_MINUTES:
            return
        
        df = pd.DataFrame(self.bars_1m)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Resample to 3-minute bars
        agg = df.resample(f"{self.AGG_TF_MINUTES}min", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        self.bars_agg = agg.reset_index()
    
    def _calculate_atr(self) -> float:
        """Calculate ATR from aggregated bars."""
        if len(self.bars_agg) < self.ATR_PERIOD + 1:
            return 0.0
        
        df = self.bars_agg.tail(self.ATR_PERIOD + 1).copy()
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(
            lambda r: max(
                r["high"] - r["low"],
                abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0
            ),
            axis=1
        )
        return df["tr"].tail(self.ATR_PERIOD).mean()
    
    @property
    def _last_price(self) -> Optional[float]:
        """Get last known price."""
        if len(self.bars_1m) > 0:
            return self.bars_1m[-1]["close"]
        return None
    
    async def run(self):
        """Main bot loop."""
        self.running = True
        
        # Connect
        await self.connect()
        await self.fetch_initial_data()
        
        # Start Telegram bot
        if self.telegram.enabled:
            await self.telegram.start()
        
        # Set start time now (after warmup complete)
        self.start_time = datetime.now(timezone.utc)
        
        logger.info("Bot started - entering main loop")
        
        last_bar_ts = self.bars_1m[-1]["timestamp"] if self.bars_1m else 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.running:
                try:
                    # Fetch latest candles
                    ohlcv = await self.exchange.fetch_ohlcv(
                        self.SYMBOL,
                        self.BASE_TF,
                        limit=5
                    )
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    
                    # Process new bars
                    for candle in ohlcv:
                        if candle[0] > last_bar_ts:
                            bar = {
                                "timestamp": candle[0],
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[5])
                            }
                            self.bars_1m.append(bar)
                            last_bar_ts = candle[0]
                            
                            # Keep only recent bars
                            if len(self.bars_1m) > 1000:
                                self.bars_1m = self.bars_1m[-800:]
                    
                    # Aggregate and check for closed 3m bar
                    old_len = len(self.bars_agg)
                    self._aggregate_bars()
                    
                    if len(self.bars_agg) > old_len:
                        # New 3m bar closed
                        await self._on_bar_close()
                    
                    # Fetch ticker for /market command (less frequently)
                    try:
                        self.cached_last_ticker = await self.exchange.fetch_ticker(self.SYMBOL)
                    except Exception:
                        pass  # Non-critical, ignore errors
                    
                    # Check position management
                    if self.position:
                        await self._manage_position()
                    
                    # AI Proactive Scan in main loop (when no position)
                    if self.position is None and self.ai_mode in ["advisory", "autonomous", "hybrid"]:
                        if self.cached_last_price and self.cached_last_atr:
                            await self._ai_proactive_scan(self.cached_last_price, self.cached_last_atr)
                    
                    # Autonomous pair switching (check for better opportunities)
                    await self._autonomous_pair_check()
                    
                    # Autonomous periodic summaries
                    await self._check_send_autonomous_summary()
                    
                    # AI self-adjustment based on performance
                    await self._ai_self_adjust()
                    
                except Exception as loop_error:
                    consecutive_errors += 1
                    logger.warning(f"⚠️ Loop error ({consecutive_errors}/{max_consecutive_errors}): {loop_error}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"❌ Too many consecutive errors, shutting down")
                        break
                    
                    # Wait longer on errors before retrying
                    await asyncio.sleep(10)
                    continue
                
                # Sleep before next iteration
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.shutdown()
    
    async def _on_bar_close(self):
        """Handle a new aggregated bar close."""
        # Update equity curve on every bar
        self._update_equity_curve()
        
        if len(self.bars_agg) < self.WARMUP_BARS:
            remaining = self.WARMUP_BARS - len(self.bars_agg)
            if remaining % 10 == 0 or remaining <= 5:
                logger.info(f"⏳ Warming up... {remaining} bars remaining")
            return
        
        current_price = self.bars_agg.iloc[-1]["close"]
        atr = self._calculate_atr()
        
        # Cache values for Telegram commands
        self.cached_last_price = current_price
        self.cached_last_atr = atr
        
        if atr == 0:
            logger.warning("ATR is 0, skipping signal check")
            return
        
        # Check if trading is paused
        if self.paused:
            return
        
        # === DAILY LOSS LIMIT CHECK (Circuit Breaker) ===
        await self._check_daily_loss_limit()
        if self.daily_loss_triggered:
            return  # Stop trading for the day
        
        # === REGIME-AWARE SIGNAL GENERATION ===
        # Use regime-aware signals that switch between trend-following and mean-reversion
        df_signals, regime_info = generate_regime_aware_signals(self.bars_agg.copy())
        signal = int(df_signals.iloc[-1].get("Side", 0))
        
        # Log every bar close with regime info
        logger.info(f"📊 Bar close: Price=${current_price:.4f} ATR=${atr:.4f} Regime={regime_info.get('regime')} Signal={signal}")
        logger.info(f"📊 Position={self.position is not None} | AI_mode={self.ai_mode}")
        
        # Log regime if it changed or on first signal
        if hasattr(self, '_last_regime') and self._last_regime != regime_info.get('regime'):
            logger.info(f"📊 Regime changed: {self._last_regime} → {regime_info.get('regime')} ({regime_info.get('strategy')})")
        self._last_regime = regime_info.get('regime')
        
        # === SMART BTC FILTER ===
        if signal != 0:
            logger.info(f"📊 Raw signal: {'LONG' if signal == 1 else 'SHORT'} | Regime: {regime_info.get('regime')} | Strategy: {regime_info.get('strategy')}")
            btc_result = smart_btc_filter(self.bars_agg, signal)
            if btc_result['should_filter']:
                logger.info(f"🔒 Signal filtered by smart BTC filter: {btc_result['reason']}")
                signal = 0
        
        # Only process if no position and we have a signal
        if self.position is None and signal != 0:
            await self._process_signal(signal, current_price, atr, regime_info)
        
        # AI Proactive Scan (if enabled and no position and no technical signal)
        if self.position is None and signal == 0 and self.ai_mode in ["advisory", "autonomous", "hybrid"]:
            logger.info(f"🔍 Triggering AI proactive scan (mode={self.ai_mode})...")
            await self._ai_proactive_scan(current_price, atr)
    
    async def _check_send_autonomous_summary(self):
        """Automatically send periodic trading summaries via Telegram."""
        if not self.summary_notifications_enabled:
            return
        
        now = datetime.now(timezone.utc)
        
        # Check for daily summary (at midnight UTC)
        today = now.date()
        if self.last_daily_summary_date != today and now.hour == 0:
            # Send daily summary
            await self._send_daily_summary()
            self.last_daily_summary_date = today
            return
        
        # Check for periodic summary (every 4 hours)
        if self.last_summary_time is None:
            self.last_summary_time = now
            return
        
        elapsed = (now - self.last_summary_time).total_seconds()
        if elapsed >= self.summary_interval:
            await self._send_periodic_summary()
            self.last_summary_time = now
    
    async def _ai_self_adjust(self):
        """AI autonomously adjusts settings based on trading performance."""
        if not self.ai_self_adjust_enabled or self.ai_mode != "autonomous":
            return
        
        now = datetime.now(timezone.utc)
        if self.last_ai_self_adjust:
            elapsed = (now - self.last_ai_self_adjust).total_seconds()
            if elapsed < self.ai_self_adjust_interval:
                return
        
        self.last_ai_self_adjust = now
        
        # Analyze recent performance
        adjustments_made = []
        
        # Check win rate and consecutive losses
        if self.stats.total_trades >= 5:
            win_rate = self.stats.win_rate
            
            # If losing streak >= 3, reduce risk
            if self.consecutive_losses >= 3:
                old_risk = self.RISK_PCT
                new_risk = max(0.01, old_risk * 0.5)  # Halve risk, min 1%
                if new_risk != old_risk:
                    self.RISK_PCT = new_risk
                    adjustments_made.append(f"📉 Risk reduced: {old_risk*100:.1f}% → {new_risk*100:.1f}% (losing streak: {self.consecutive_losses})")
                    logger.info(f"🤖 AI SELF-ADJUST: Reduced risk to {new_risk*100:.1f}% due to {self.consecutive_losses} losses")
            
            # If winning streak >= 3 and good win rate, slightly increase risk
            elif self.consecutive_wins >= 3 and win_rate > 0.55:
                old_risk = self.RISK_PCT
                new_risk = min(0.04, old_risk * 1.25)  # Increase by 25%, max 4%
                if new_risk != old_risk:
                    self.RISK_PCT = new_risk
                    adjustments_made.append(f"📈 Risk increased: {old_risk*100:.1f}% → {new_risk*100:.1f}% (winning streak: {self.consecutive_wins})")
                    logger.info(f"🤖 AI SELF-ADJUST: Increased risk to {new_risk*100:.1f}% due to {self.consecutive_wins} wins")
            
            # If win rate drops below 40%, tighten AI confidence threshold
            if win_rate < 0.40 and self.ai_filter.confidence_threshold < 0.85:
                old_conf = self.ai_filter.confidence_threshold
                new_conf = min(0.85, old_conf + 0.05)
                self.ai_filter.confidence_threshold = new_conf
                adjustments_made.append(f"🎯 AI confidence raised: {old_conf*100:.0f}% → {new_conf*100:.0f}% (win rate: {win_rate*100:.0f}%)")
                logger.info(f"🤖 AI SELF-ADJUST: Raised confidence threshold to {new_conf*100:.0f}%")
            
            # If win rate above 60%, can relax confidence slightly
            elif win_rate > 0.60 and self.ai_filter.confidence_threshold > 0.65:
                old_conf = self.ai_filter.confidence_threshold
                new_conf = max(0.65, old_conf - 0.05)
                self.ai_filter.confidence_threshold = new_conf
                adjustments_made.append(f"🎯 AI confidence relaxed: {old_conf*100:.0f}% → {new_conf*100:.0f}% (win rate: {win_rate*100:.0f}%)")
                logger.info(f"🤖 AI SELF-ADJUST: Relaxed confidence threshold to {new_conf*100:.0f}%")
        
        # Notify about adjustments
        if adjustments_made and self.telegram.enabled:
            msg = "🤖 *AI AUTONOMOUS ADJUSTMENT*\n\n"
            msg += "\n".join(adjustments_made)
            msg += f"\n\n_Current: Risk={self.RISK_PCT*100:.1f}%, Confidence={self.ai_filter.confidence_threshold*100:.0f}%_"
            await self.telegram.send_message(msg)
            
            # Log the decision
            self.ai_decisions_log.append({
                "time": now.isoformat(),
                "type": "self_adjust",
                "adjustments": adjustments_made
            })
    
    async def _notify_ai_decision(self, decision_type: str, details: Dict[str, Any]):
        """Notify user about AI decision with full transparency."""
        now = datetime.now(timezone.utc)
        
        # Log the decision
        self.ai_decisions_log.append({
            "time": now.isoformat(),
            "type": decision_type,
            "details": details
        })
        
        # Keep only last 100 decisions
        if len(self.ai_decisions_log) > 100:
            self.ai_decisions_log = self.ai_decisions_log[-100:]
        
        # Always notify for important decisions
        if self.telegram.enabled:
            if decision_type == "scan_result":
                if details.get("opportunity"):
                    opp = details["opportunity"]
                    msg = f"""🔍 *AI SCAN RESULT*

📊 *{self.SYMBOL}* @ ${details.get('price', 0):,.4f}
🎯 *Decision:* {opp.get('action', 'WAIT')}
📈 *Confidence:* `{opp.get('confidence', 0)*100:.0f}%`
💡 *Reasoning:* {opp.get('reasoning', 'N/A')[:200]}
⚠️ *Risk Level:* {opp.get('risk_assessment', 'medium')}

_AI Mode: {self.ai_mode}_"""
                    await self.telegram.send_message(msg)
                else:
                    # Skip "no opportunity" notifications if configured
                    if self.ai_scan_notify_opportunities_only:
                        logger.debug("AI scan: No opportunity (notification suppressed)")
                        return
                    
                    # Rate limit "no opportunity" notifications
                    if self.last_ai_decision_notification:
                        elapsed = (now - self.last_ai_decision_notification).total_seconds()
                        if elapsed < self.ai_scan_quiet_interval:
                            return
                    self.last_ai_decision_notification = now
                    
                    msg = f"""🔍 *AI SCAN*

📊 *{self.SYMBOL}* @ ${details.get('price', 0):,.4f}
🎯 *Decision:* No trade opportunity
💭 *Status:* Watching market...

_Next scan in ~{self.ai_scan_interval//60} min_"""
                    await self.telegram.send_message(msg)
            
            elif decision_type == "trade_opened":
                msg = f"""🚀 *AI TRADE OPENED*

📊 *{details.get('symbol', self.SYMBOL)}*
💰 *Side:* `{details.get('side', 'N/A')}`
📈 *Entry:* `${details.get('price', 0):,.4f}`
🎯 *Confidence:* `{details.get('confidence', 0)*100:.0f}%`
💵 *Risk:* `{details.get('risk_pct', 0)*100:.1f}%` (${details.get('risk_amount', 0):,.2f})
💡 *Reason:* {details.get('reasoning', 'AI Autonomous')}

_Executed autonomously by AI_"""
                await self.telegram.send_message(msg)
            
            elif decision_type == "trade_closed":
                emoji = "✅" if details.get('pnl', 0) >= 0 else "❌"
                msg = f"""{emoji} *AI TRADE CLOSED*

📊 *{details.get('symbol', self.SYMBOL)}*
💰 *Side:* `{details.get('side', 'N/A')}`
📈 *Exit:* `${details.get('price', 0):,.4f}`
💵 *P&L:* `${details.get('pnl', 0):+,.2f}` ({details.get('pnl_pct', 0):+.2f}%)
📝 *Reason:* {details.get('reason', 'N/A')}

_Balance: ${self.balance:,.2f}_"""
                await self.telegram.send_message(msg)
            
            elif decision_type == "signal_filtered":
                msg = f"""🛡️ *AI SIGNAL FILTERED*

📊 *{self.SYMBOL}* @ ${details.get('price', 0):,.4f}
⚠️ *Original Signal:* `{details.get('signal', 'N/A')}`
❌ *AI Decision:* REJECT
📈 *AI Confidence:* `{details.get('confidence', 0)*100:.0f}%`
💭 *Reason:* {details.get('reasoning', 'N/A')[:200]}

_Technical signal filtered by AI_"""
                await self.telegram.send_message(msg)

    async def _check_daily_loss_limit(self):
        """Check and enforce daily loss limit (circuit breaker)."""
        today = datetime.now(timezone.utc).date()
        
        # Reset circuit breaker at midnight UTC
        if self.daily_loss_reset_date != today:
            self.daily_loss_reset_date = today
            if self.daily_loss_triggered:
                logger.info("🔄 Daily loss circuit breaker reset (new day)")
                self.daily_loss_triggered = False
                if self.telegram.enabled:
                    await self.telegram.send_message("🔄 *Daily Loss Limit Reset*\nTrading resumed for new day.")
        
        # Check if daily loss exceeds limit
        if self.stats.today_pnl < 0:
            daily_loss_pct = abs(self.stats.today_pnl) / self.initial_balance
            
            if daily_loss_pct >= self.daily_loss_limit and not self.daily_loss_triggered:
                self.daily_loss_triggered = True
                logger.warning(f"🛑 DAILY LOSS LIMIT HIT: {daily_loss_pct*100:.1f}% (limit: {self.daily_loss_limit*100:.1f}%)")
                logger.warning("Trading halted until tomorrow or manual reset")
                
                if self.telegram.enabled:
                    await self.telegram.send_message(
                        f"🛑 *DAILY LOSS LIMIT TRIGGERED*\n\n"
                        f"Daily Loss: `${self.stats.today_pnl:.2f}` ({daily_loss_pct*100:.1f}%)\n"
                        f"Limit: `{self.daily_loss_limit*100:.1f}%`\n\n"
                        f"_Trading halted until tomorrow._\n"
                        f"Use `/set reset_daily_loss true` to override."
                    )
    
    async def _send_periodic_summary(self):
        """Send a periodic status summary."""
        if not self.telegram.enabled:
            return
        
        # Calculate stats
        uptime = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        
        pnl_pct = ((self.balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        drawdown = ((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
        
        position_status = "None"
        if self.position:
            unrealized = self.position.unrealized_pnl(self.cached_last_price or self.position.entry_price)
            position_status = f"{self.position.side.upper()} (${unrealized:+.2f})"
        
        # Get ML status
        try:
            ml_classifier = get_ml_classifier()
            ml_stats = ml_classifier.get_stats()
            ml_status = f"Trained ({ml_stats.get('total_samples', 0)} samples)" if ml_stats.get('is_trained') else f"Learning ({ml_stats.get('samples_until_training', 50)} needed)"
        except:
            ml_status = "N/A"
        
        emoji = "📈" if self.stats.total_pnl >= 0 else "📉"
        
        # Add circuit breaker status
        circuit_status = "🛑 HALTED" if self.daily_loss_triggered else "✅ Active"
        dry_run_status = "📝 DRY-RUN" if self.dry_run_mode else ""
        
        msg = f"""
🤖 *Julaba Status Update* {dry_run_status}

⏱ *Uptime:* `{hours}h {minutes}m`
💰 *Balance:* `${self.balance:,.2f}` ({pnl_pct:+.2f}%)
{emoji} *Total P&L:* `${self.stats.total_pnl:+,.2f}`
📊 *Drawdown:* `{drawdown:.1f}%`
🚦 *Circuit:* `{circuit_status}`

*Trading Stats*
🎯 Trades: `{self.stats.total_trades}` ({self.stats.winning_trades}W / {self.stats.losing_trades}L)
📈 Win Rate: `{self.stats.win_rate * 100:.1f}%`
🔥 Streak: `{self.consecutive_wins}W / {self.consecutive_losses}L`

*Current State*
📍 Position: `{position_status}`
💵 Price: `${self.cached_last_price:,.4f}`
🧠 ML: `{ml_status}`
🤖 AI Mode: `{self.ai_mode}`

_Auto-update every 4 hours_
"""
        await self.telegram.send_message(msg)
        logger.info("Periodic summary sent to Telegram")
    
    async def _send_daily_summary(self):
        """Send daily trading summary at midnight UTC."""
        if not self.telegram.enabled:
            return
        
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Daily P&L from today_pnl stat
        today_pnl = self.stats.today_pnl
        
        await self.telegram.notify_daily_summary(
            date=yesterday,
            trades=self.stats.total_trades,
            wins=self.stats.winning_trades,
            losses=self.stats.losing_trades,
            pnl=today_pnl,
            balance=self.balance,
            win_rate=self.stats.win_rate * 100
        )
        
        # Reset daily stats
        self.stats.today_pnl = 0.0
        logger.info(f"Daily summary sent for {yesterday}")

    async def _ai_proactive_scan(self, price: float, atr: float):
        """Let AI proactively scan for opportunities."""
        # Rate limit: only scan based on ai_scan_interval
        now = datetime.now(timezone.utc)
        if self.last_ai_scan_time:
            elapsed = (now - self.last_ai_scan_time).total_seconds()
            if elapsed < self.ai_scan_interval:
                logger.debug(f"AI scan rate-limited: {elapsed:.0f}s / {self.ai_scan_interval}s")
                return
        
        self.last_ai_scan_time = now
        logger.info(f"🔍 AI Proactive Scan starting... (interval: {self.ai_scan_interval}s)")
        
        # AI scans the market
        opportunity = self.ai_filter.proactive_scan(
            df=self.bars_agg,
            current_price=price,
            atr=atr,
            symbol=self.SYMBOL
        )
        
        logger.info(f"🔍 AI Scan result: {'Found opportunity!' if opportunity else 'No opportunity'}")
        
        # Notify about scan result (opportunity or not)
        await self._notify_ai_decision("scan_result", {
            "price": price,
            "opportunity": opportunity
        })
        
        if not opportunity:
            return
        
        # Handle based on AI mode
        if self.ai_mode == "autonomous":
            # AI opens trade directly with full autonomy (70%+ confidence)
            ai_risk_pct = opportunity.get('suggested_risk_pct', self.RISK_PCT)
            risk_level = opportunity.get('risk_assessment', 'medium')
            
            # AI can adjust risk based on its assessment
            if risk_level == "low":
                ai_risk_pct = min(ai_risk_pct * 1.5, 0.05)  # Up to 5% on low risk
            elif risk_level == "high":
                ai_risk_pct = ai_risk_pct * 0.5  # Reduce on high risk
            
            logger.info(f"🤖 AI AUTONOMOUS: Opening {opportunity['action']} | Risk: {ai_risk_pct:.1%} ({risk_level})")
            
            # Notify about trade opening
            risk_amount = self.balance * ai_risk_pct
            await self._notify_ai_decision("trade_opened", {
                "symbol": self.SYMBOL,
                "side": opportunity['action'],
                "price": price,
                "confidence": opportunity['confidence'],
                "risk_pct": ai_risk_pct,
                "risk_amount": risk_amount,
                "reasoning": opportunity['reasoning']
            })
            
            # Store AI's risk preference temporarily
            original_risk = self.RISK_PCT
            self.RISK_PCT = ai_risk_pct
            
            await self._open_position(
                opportunity['signal'],
                price,
                atr,
                source="ai_autonomous"
            )
            
            # Restore original risk setting
            self.RISK_PCT = original_risk
            
            if self.telegram.enabled:
                await self.telegram.notify_ai_trade(
                    symbol=self.SYMBOL,
                    action=opportunity['action'],
                    price=price,
                    confidence=opportunity['confidence'],
                    reasoning=opportunity['reasoning'],
                    mode="autonomous"
                )
        
        elif self.ai_mode in ["advisory", "hybrid"]:
            # Store pending trade and ask user for confirmation
            self.pending_ai_trade = opportunity
            logger.info(f"🤖 AI {self.ai_mode.upper()}: Suggesting {opportunity['action']} - awaiting confirmation")
            
            if self.telegram.enabled:
                await self.telegram.notify_ai_trade(
                    symbol=self.SYMBOL,
                    action=opportunity['action'],
                    price=price,
                    confidence=opportunity['confidence'],
                    reasoning=opportunity['reasoning'],
                    mode=self.ai_mode
                )

    async def _check_btc_crash_protection(self) -> Dict[str, Any]:
        """Check if BTC has crashed and activate cooldown if needed."""
        result = {"cooldown_active": False, "btc_change": 0, "reason": None}
        
        try:
            # Get current BTC price
            ticker = await self.exchange.fetch_ticker("BTC/USDT")
            current_btc = ticker['last']
            
            # Initialize if first check
            if self.last_btc_price is None:
                self.last_btc_price = current_btc
                return result
            
            # Calculate BTC change
            btc_change = (current_btc - self.last_btc_price) / self.last_btc_price
            result['btc_change'] = btc_change
            
            # Check for crash
            if btc_change <= self.btc_crash_threshold:
                self.btc_crash_cooldown = True
                self.btc_crash_cooldown_until = datetime.utcnow() + timedelta(minutes=self.btc_crash_cooldown_minutes)
                result['cooldown_active'] = True
                result['reason'] = f"BTC crashed {btc_change:.1%} - activating {self.btc_crash_cooldown_minutes}min cooldown"
                logger.warning(f"🚨 {result['reason']}")
                
                if self.telegram.enabled:
                    await self.telegram.send_message(
                        f"🚨 *BTC CRASH PROTECTION ACTIVATED*\n\n"
                        f"BTC dropped: `{btc_change:.1%}`\n"
                        f"Cooldown: `{self.btc_crash_cooldown_minutes} minutes`\n"
                        f"Until: `{self.btc_crash_cooldown_until.strftime('%H:%M:%S')}`\n\n"
                        f"_All new trades paused during crash correlation spike_"
                    )
            
            # Update last price periodically (every 5 minutes)
            if btc_change > 0.02 or btc_change < -0.02:  # >2% move
                self.last_btc_price = current_btc
            
            # Check if cooldown expired
            if self.btc_crash_cooldown and datetime.utcnow() > self.btc_crash_cooldown_until:
                self.btc_crash_cooldown = False
                logger.info("✅ BTC crash cooldown expired - trading resumed")
                if self.telegram.enabled:
                    await self.telegram.send_message("✅ *BTC crash cooldown expired* - Trading resumed")
            
            if self.btc_crash_cooldown:
                result['cooldown_active'] = True
                result['reason'] = f"BTC crash cooldown active until {self.btc_crash_cooldown_until.strftime('%H:%M:%S')}"
                
        except Exception as e:
            logger.debug(f"BTC crash check error: {e}")
        
        return result

    def _calculate_technical_score(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a technical signal quality score (0-100).
        
        Factors:
        - ADX strength (0-30 points): Higher ADX = stronger trend
        - Hurst exponent (0-20 points): >0.5 = trending, <0.5 = mean-reverting
        - Volume ratio (0-20 points): Higher volume = more conviction
        - RSI position (0-15 points): Not overbought/oversold
        - Regime clarity (0-15 points): Clear trend or clear ranging
        """
        score = 0
        factors = []
        
        try:
            # Get indicators from bars
            df = self.bars_agg
            adx = regime_info.get('adx', 0) if regime_info else 0
            hurst = regime_info.get('hurst', 0.5) if regime_info else 0.5
            regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
            
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
            
            # ADX Score (0-30): Trend strength
            if adx >= 40:
                adx_score = 30
                factors.append(f"Strong ADX ({adx:.0f})")
            elif adx >= 30:
                adx_score = 25
            elif adx >= 25:
                adx_score = 20
            elif adx >= 20:
                adx_score = 10
            else:
                adx_score = 5
                factors.append(f"Weak ADX ({adx:.0f})")
            score += adx_score
            
            # Hurst Score (0-20): Trendiness
            if hurst >= 0.6:
                hurst_score = 20
                factors.append(f"Trending (H={hurst:.2f})")
            elif hurst >= 0.5:
                hurst_score = 15
            elif hurst >= 0.4:
                hurst_score = 10
            else:
                hurst_score = 5
                factors.append(f"Choppy (H={hurst:.2f})")
            score += hurst_score
            
            # Volume Score (0-20): Conviction
            if volume_ratio >= 2.0:
                vol_score = 20
                factors.append(f"High volume ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.5:
                vol_score = 15
            elif volume_ratio >= 1.0:
                vol_score = 10
            elif volume_ratio >= 0.5:
                vol_score = 5
            else:
                vol_score = 0
                factors.append(f"Low volume ({volume_ratio:.1f}x)")
            score += vol_score
            
            # RSI Score (0-15): Not at extremes
            if 40 <= rsi <= 60:
                rsi_score = 15
            elif 30 <= rsi <= 70:
                rsi_score = 10
            elif 20 <= rsi <= 80:
                rsi_score = 5
            else:
                rsi_score = 0
                factors.append(f"RSI extreme ({rsi:.0f})")
            score += rsi_score
            
            # Regime Score (0-15): Clear market state
            if regime in ['TRENDING', 'STRONG_TREND']:
                regime_score = 15
            elif regime == 'RANGING':
                regime_score = 10
            elif regime == 'WEAK_TREND':
                regime_score = 5
            else:
                regime_score = 0
                factors.append(f"Unclear regime")
            score += regime_score
            
        except Exception as e:
            logger.debug(f"Technical score error: {e}")
            score = 50  # Default middle score
            factors.append("Error calculating")
        
        # Quality label
        if score >= 80:
            quality = "EXCELLENT"
        elif score >= 65:
            quality = "GOOD"
        elif score >= 50:
            quality = "MODERATE"
        elif score >= 35:
            quality = "WEAK"
        else:
            quality = "POOR"
        
        return {
            'score': score,
            'quality': quality,
            'factors': factors,
            'breakdown': {
                'adx': adx_score if 'adx_score' in dir() else 0,
                'hurst': hurst_score if 'hurst_score' in dir() else 0,
                'volume': vol_score if 'vol_score' in dir() else 0,
                'rsi': rsi_score if 'rsi_score' in dir() else 0,
                'regime': regime_score if 'regime_score' in dir() else 0
            }
        }

    def _calculate_system_score(
        self,
        tech_score: Dict[str, Any],
        ml_result: Dict[str, Any],
        regime_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate combined system score (0-100).
        
        Weights:
        - Technical score: 50%
        - ML prediction: 30%
        - Regime alignment: 20%
        
        This gives AI a single "system confidence" number.
        """
        # Technical component (0-50)
        tech_component = tech_score['score'] * 0.50
        
        # ML component (0-30)
        if ml_result.get('ml_available'):
            # Convert probability (0.3-0.7 range typically) to 0-100 scale
            ml_prob = ml_result['ml_win_probability']
            # Scale: 0.40=0, 0.50=50, 0.60=100
            ml_scaled = min(100, max(0, (ml_prob - 0.40) * 500))
            ml_component = ml_scaled * 0.30
        else:
            ml_component = 15  # Neutral if ML not available
        
        # Regime alignment component (0-20)
        regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
        strategy = regime_info.get('strategy', 'NONE') if regime_info else 'NONE'
        
        if regime in ['TRENDING', 'STRONG_TREND'] and strategy == 'TREND_FOLLOWING':
            regime_component = 20  # Perfect alignment
        elif regime == 'RANGING' and strategy == 'MEAN_REVERSION':
            regime_component = 18  # Good alignment
        elif regime == 'WEAK_TREND':
            regime_component = 10  # Acceptable
        elif regime == 'CHOPPY' or strategy == 'NO_TRADE':
            regime_component = 0  # Should not trade
        else:
            regime_component = 10  # Unknown = neutral
        
        combined = tech_component + ml_component + regime_component
        
        # Recommendation based on combined score
        if combined >= 75:
            recommendation = "STRONG_BUY"
        elif combined >= 60:
            recommendation = "BUY"
        elif combined >= 45:
            recommendation = "NEUTRAL"
        elif combined >= 30:
            recommendation = "WEAK"
        else:
            recommendation = "AVOID"
        
        return {
            'combined': combined,
            'recommendation': recommendation,
            'tech_component': tech_component,
            'ml_component': ml_component,
            'regime_component': regime_component,
            'breakdown': f"Tech:{tech_component:.0f} + ML:{ml_component:.0f} + Regime:{regime_component:.0f}"
        }

    def _log_decision_alignment(
        self,
        ml_result: Dict[str, Any],
        ai_result: Dict[str, Any],
        system_score: Dict[str, Any]
    ):
        """Log how different system components align on the decision."""
        ml_available = ml_result.get('ml_available', False)
        ml_favorable = ml_result.get('ml_win_probability', 0.5) >= 0.55 if ml_available else None
        ai_approved = ai_result.get('approved', False)
        system_favorable = system_score['combined'] >= 60
        
        if not ml_available:
            if ai_approved:
                logger.info(f"🤖 AI APPROVED (ML not available) | System: {system_score['combined']:.0f}")
            else:
                logger.info(f"🤖 AI REJECTED (ML not available) | System: {system_score['combined']:.0f}")
            return
        
        # All three align
        if ml_favorable == ai_approved == system_favorable:
            if ai_approved:
                logger.info(f"✅ FULL ALIGNMENT: ML+AI+System all approve ({system_score['combined']:.0f}/100)")
            else:
                logger.info(f"⛔ FULL ALIGNMENT: ML+AI+System all reject ({system_score['combined']:.0f}/100)")
        # AI overrides
        elif ai_approved and not ml_favorable:
            logger.info(f"🤖 AI OVERRIDE: Approved despite weak ML ({ml_result['ml_win_probability']:.1%})")
        elif not ai_approved and ml_favorable:
            logger.info(f"🤖 AI VETO: Rejected despite favorable ML ({ml_result['ml_win_probability']:.1%})")
        # Mixed signals
        else:
            logger.info(f"⚠️ MIXED: AI={ai_approved}, ML={ml_favorable}, System={system_favorable}")

    async def _process_signal(self, signal: int, price: float, atr: float, regime_info: Dict[str, Any] = None):
        """
        Process a trading signal through the complete decision pipeline.
        
        DECISION FLOW (Signal → ML → AI):
        1. MATH SIGNAL: Technical indicators generate raw signal (+1/-1)
        2. ML SCORING: XGBoost predicts win probability based on historical patterns
        3. AI DECISION: Gemini AI makes FINAL decision with all context
        
        The AI sees:
        - Technical signal strength and regime
        - ML prediction and confidence
        - Combined system score
        - Trading performance history
        """
        side = "LONG" if signal == 1 else "SHORT"
        strategy = regime_info.get('strategy', 'TREND_FOLLOWING') if regime_info else 'TREND_FOLLOWING'
        
        # === PHASE 0: BTC CRASH PROTECTION (System Override) ===
        btc_status = await self._check_btc_crash_protection()
        if btc_status['cooldown_active']:
            logger.info(f"🚫 Signal {side} BLOCKED: {btc_status['reason']}")
            return
        
        # === PHASE 1: CALCULATE TECHNICAL SCORE ===
        tech_score = self._calculate_technical_score(regime_info)
        logger.info(f"📈 Technical Score: {tech_score['score']:.0f}/100 ({tech_score['quality']})")
        
        # === PHASE 2: ML PREDICTION ===
        ml_result = {'ml_available': False, 'ml_win_probability': 0.5}
        ml_insight = {'ml_available': False}
        
        if self.ml_predictor.is_loaded:
            # Build feature dict from current market state
            ml_features = {
                'atr_percent': (atr / price) * 100 if price > 0 else 0,
                'rsi': self.bars_agg['rsi'].iloc[-1] if 'rsi' in self.bars_agg.columns else 50,
                'adx': self.bars_agg['adx'].iloc[-1] if 'adx' in self.bars_agg.columns else 0,
                'volume_ratio': self.bars_agg['volume_ratio'].iloc[-1] if 'volume_ratio' in self.bars_agg.columns else 1,
                'hurst': self.bars_agg['hurst'].iloc[-1] if 'hurst' in self.bars_agg.columns else 0.5,
                'sma_distance_percent': 0,
                'hour': datetime.utcnow().hour,
                'day_of_week': datetime.utcnow().weekday(),
                'regime': regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
            }
            ml_result = self.ml_predictor.predict(ml_features)
            
            if ml_result.get('ml_available'):
                logger.info(f"🧠 ML Score: {ml_result['ml_win_probability']:.1%} win prob ({ml_result['ml_confidence']})")
                ml_insight = {
                    'ml_available': True,
                    'ml_win_probability': ml_result['ml_win_probability'],
                    'ml_confidence': ml_result['ml_confidence'],
                    'ml_accuracy': self.ml_predictor.model_metadata.get('accuracy', 0.5),
                    'ml_samples': self.ml_predictor.model_metadata.get('train_samples', 0),
                    'ml_influence': 0.0  # Currently advisory only
                }
        
        # === PHASE 3: CALCULATE COMBINED SYSTEM SCORE ===
        system_score = self._calculate_system_score(
            tech_score=tech_score,
            ml_result=ml_result,
            regime_info=regime_info
        )
        logger.info(f"🎯 System Score: {system_score['combined']:.0f}/100 ({system_score['recommendation']})")
        
        # === PHASE 4: AI FINAL DECISION (with complete context) ===
        ai_result = self.ai_filter.analyze_signal(
            signal=signal,
            df=self.bars_agg,
            current_price=price,
            atr=atr,
            symbol=self.SYMBOL,
            ml_insight=ml_insight,
            system_score=system_score  # NEW: Pass combined system score
        )
        
        # Log decision alignment
        self._log_decision_alignment(ml_result, ai_result, system_score)
        
        # Record signal in history (complete data for analysis)
        self.signal_history.append({
            "side": side,
            "price": price,
            "approved": ai_result["approved"],
            "confidence": ai_result["confidence"],
            "strategy": strategy,
            "regime": regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN',
            "tech_score": tech_score['score'],
            "ml_probability": ml_result.get('ml_win_probability', 0.5),
            "ml_confidence": ml_result.get('ml_confidence', 'N/A'),
            "system_score": system_score['combined'],
            "time": datetime.utcnow().strftime("%H:%M:%S")
        })
        # Keep only last 50 signals
        if len(self.signal_history) > 50:
            self.signal_history = self.signal_history[-50:]
        
        # Notify via Telegram
        if self.telegram.enabled:
            await self.telegram.notify_signal(
                symbol=self.SYMBOL,
                side=side,
                price=price,
                ai_approved=ai_result["approved"],
                confidence=ai_result["confidence"],
                reasoning=ai_result["reasoning"]
            )
        
        # Execute if approved
        if ai_result["approved"]:
            await self._open_position(signal, price, atr)
        else:
            logger.info(f"Signal {side} REJECTED by AI filter: {ai_result['reasoning']}")
            # Notify about AI filtering decision
            await self._notify_ai_decision("signal_filtered", {
                "signal": side,
                "price": price,
                "confidence": ai_result["confidence"],
                "reasoning": ai_result["reasoning"]
            })
    
    async def _open_position(self, signal: int, price: float, atr: float, source: str = "technical"):
        """Open a new position with intelligent risk management."""
        side = "long" if signal == 1 else "short"
        
        # Intelligent pattern detection
        pattern = detect_candlestick_patterns(self.bars_agg) if len(self.bars_agg) >= 3 else {}
        
        # ML regime prediction
        ml_prediction = ml_predict_regime(self.bars_agg) if len(self.bars_agg) >= 50 else {}
        
        # Smart drawdown-adjusted risk
        drawdown_info = calculate_drawdown_adjusted_risk(
            base_risk=self.RISK_PCT,
            current_balance=self.balance,
            peak_balance=self.peak_balance,
            consecutive_losses=self.consecutive_losses,
            consecutive_wins=self.consecutive_wins
        )
        
        # Use adjusted risk for position sizing
        adjusted_risk = drawdown_info['adjusted_risk']
        
        # Log intelligence
        if pattern.get('pattern'):
            logger.info(f"📊 Pattern: {pattern['pattern']} ({'Bullish' if pattern.get('bullish') else 'Bearish'})")
        if ml_prediction.get('is_trained'):
            logger.info(f"🧠 ML: {ml_prediction.get('ml_signal', 'N/A')} ({ml_prediction.get('ml_score', 0):.0%})")
        logger.info(f"🎯 Risk Mode: {drawdown_info['mode']} ({adjusted_risk:.1%} risk)")
        
        # Calculate stop loss
        if side == "long":
            stop_loss = price - (atr * self.ATR_MULT)
        else:
            stop_loss = price + (atr * self.ATR_MULT)
        
        risk_per_unit = abs(price - stop_loss)
        risk_amount = self.balance * adjusted_risk  # Use adjusted risk
        
        # === REALISTIC COST MODEL ===
        # Account for slippage + fees in position sizing
        # Entry: price + slippage + fee
        # Exit: SL/TP - slippage + fee
        # Total cost = 2 * (slippage + fee) = ~0.6% of position
        effective_entry_price = price * (1 + self.SLIPPAGE_PCT) if side == "long" else price * (1 - self.SLIPPAGE_PCT)
        fee_cost_pct = self.FEE_TAKER * 2  # Open + close
        
        # Adjust size to account for costs
        # Net risk = (entry - SL) - (entry * fee_cost)
        cost_adjusted_risk_per_unit = risk_per_unit * (1 - fee_cost_pct)
        size = risk_amount / cost_adjusted_risk_per_unit if cost_adjusted_risk_per_unit > 0 else 0
        
        logger.debug(f"Position sizing: Risk ${risk_amount:.2f}, Cost-adj R: ${cost_adjusted_risk_per_unit:.4f}, Size: {size:.4f}")
        
        # Calculate take profits
        r_value = risk_per_unit
        if side == "long":
            tp1 = price + (r_value * self.TP1_R)
            tp2 = price + (r_value * self.TP2_R)
            tp3 = price + (r_value * self.TP3_R)
        else:
            tp1 = price - (r_value * self.TP1_R)
            tp2 = price - (r_value * self.TP2_R)
            tp3 = price - (r_value * self.TP3_R)
        
        # Store entry snapshot for ML learning
        entry_snapshot = self.bars_agg.copy() if len(self.bars_agg) >= 50 else None
        
        # === DRY-RUN MODE: Log but don't execute ===
        if self.dry_run_mode:
            logger.info(f"📝 [DRY-RUN] Would open {side.upper()} @ {price:.4f} | Size: {size:.4f} | SL: {stop_loss:.4f}")
            if self.telegram.enabled:
                await self.telegram.send_message(
                    f"📝 *DRY-RUN SIGNAL*\n\n"
                    f"Side: `{side.upper()}`\n"
                    f"Entry: `${price:.4f}`\n"
                    f"Size: `{size:.4f}`\n"
                    f"SL: `${stop_loss:.4f}`\n"
                    f"TP1: `${tp1:.4f}`\n"
                    f"TP2: `${tp2:.4f}`\n"
                    f"TP3: `${tp3:.4f}`\n"
                    f"\n_Trade NOT executed (dry-run mode)_"
                )
            return  # Don't actually open position
        
        self.position = Position(
            symbol=self.SYMBOL,
            side=side,
            entry_price=price,
            size=size,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            entry_df_snapshot=entry_snapshot
        )
        
        source_label = "🤖 AI" if "ai" in source else "📊 Technical"
        logger.info(
            f"OPENED {side.upper()} [{source_label}] | Entry: {price:.4f} | Size: {size:.4f} | "
            f"SL: {stop_loss:.4f} | TP1: {tp1:.4f} | TP2: {tp2:.4f} | TP3: {tp3:.4f}"
        )
        
        if self.telegram.enabled:
            await self.telegram.notify_trade_opened(
                symbol=self.SYMBOL,
                side=side.upper(),
                entry_price=price,
                size=size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3
            )
    
    async def _manage_position(self):
        """Manage open position - check TP/SL levels."""
        if not self.position:
            return
        
        price = self._last_price
        if not price:
            return
        
        pos = self.position
        
        # Check stop loss
        if (pos.side == "long" and price <= pos.stop_loss) or \
           (pos.side == "short" and price >= pos.stop_loss):
            await self._close_position("Stop Loss", price)
            return
        
        # Check trailing stop
        if pos.trailing_stop:
            if (pos.side == "long" and price <= pos.trailing_stop) or \
               (pos.side == "short" and price >= pos.trailing_stop):
                await self._close_position("Trailing Stop", price)
                return
        
        # Check take profits
        if not pos.tp1_hit:
            if (pos.side == "long" and price >= pos.tp1) or \
               (pos.side == "short" and price <= pos.tp1):
                await self._hit_tp(1, price)
        
        if not pos.tp2_hit and pos.tp1_hit:
            if (pos.side == "long" and price >= pos.tp2) or \
               (pos.side == "short" and price <= pos.tp2):
                await self._hit_tp(2, price)
        
        if not pos.tp3_hit and pos.tp2_hit:
            if (pos.side == "long" and price >= pos.tp3) or \
               (pos.side == "short" and price <= pos.tp3):
                await self._hit_tp(3, price)
                await self._close_position("TP3 Hit", price)
    
    async def _hit_tp(self, level: int, price: float):
        """Handle take profit hit."""
        pos = self.position
        
        if level == 1:
            pos.tp1_hit = True
            pct = self.TP1_PCT
            # Activate trailing stop
            r_value = abs(pos.entry_price - pos.stop_loss) / self.ATR_MULT
            if pos.side == "long":
                pos.trailing_stop = price - (r_value * self.TRAIL_OFFSET_R)
            else:
                pos.trailing_stop = price + (r_value * self.TRAIL_OFFSET_R)
            logger.info(f"Trailing stop activated at {pos.trailing_stop:.4f}")
        elif level == 2:
            pos.tp2_hit = True
            pct = self.TP2_PCT
        else:
            pos.tp3_hit = True
            pct = self.TP3_PCT
        
        # Calculate P&L for this portion
        portion_size = pos.size * pct
        if pos.side == "long":
            pnl = (price - pos.entry_price) * portion_size
        else:
            pnl = (pos.entry_price - price) * portion_size
        
        self.balance += pnl
        self.stats.total_pnl += pnl
        self.stats.today_pnl += pnl
        
        remaining = 1.0 - (self.TP1_PCT if pos.tp1_hit else 0) - \
                         (self.TP2_PCT if pos.tp2_hit else 0) - \
                         (self.TP3_PCT if pos.tp3_hit else 0)
        
        logger.info(f"TP{level} HIT | Price: {price:.4f} | P&L: ${pnl:+.2f} | Remaining: {remaining:.0%}")
        
        if self.telegram.enabled:
            await self.telegram.notify_tp_hit(
                symbol=self.SYMBOL,
                tp_level=level,
                price=price,
                pnl=pnl,
                remaining_pct=remaining
            )
    
    async def _close_position(self, reason: str, price: float):
        """Close the position completely with ML learning."""
        pos = self.position
        
        # Calculate final P&L on remaining
        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.remaining_size
        else:
            pnl = (pos.entry_price - price) * pos.remaining_size
        
        self.balance += pnl
        self.stats.total_pnl += pnl
        self.stats.today_pnl += pnl
        self.stats.total_trades += 1
        
        # Update peak balance for drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Track max win/loss
        if pnl > self.stats.max_win:
            self.stats.max_win = pnl
        if pnl < self.stats.max_loss:
            self.stats.max_loss = pnl
        
        total_pnl = pnl  # This is just remaining, full P&L was already added in _hit_tp
        pnl_pct = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        is_win = pnl >= 0
        if is_win:
            self.stats.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.stats.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # ML Learning: record trade outcome for model training
        if pos.entry_df_snapshot is not None:
            ml_record_trade(pos.entry_df_snapshot, is_win)
            logger.debug(f"ML sample recorded: {'WIN' if is_win else 'LOSS'}")
        
        # Record in trade history
        self.trade_history.append({
            "side": pos.side.upper(),
            "entry": pos.entry_price,
            "exit": price,
            "pnl": pnl,
            "reason": reason,
            "time": datetime.utcnow().strftime("%H:%M:%S")
        })
        # Keep only last 50 trades
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]
        
        # Record result for AI filter learning
        self.ai_filter.record_trade_result(is_win, pnl)
        
        logger.info(
            f"CLOSED {pos.side.upper()} | Reason: {reason} | Price: {price:.4f} | "
            f"P&L: ${pnl:+.2f} | Balance: ${self.balance:,.2f} | "
            f"Streak: W{self.consecutive_wins}/L{self.consecutive_losses}"
        )
        
        if self.telegram.enabled:
            if "Stop" in reason:
                await self.telegram.notify_stop_loss(
                    symbol=self.SYMBOL,
                    price=price,
                    pnl=pnl
                )
            else:
                await self.telegram.notify_trade_closed(
                    symbol=self.SYMBOL,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason
                )
            
            # AI decision notification for trade close
            await self._notify_ai_decision("trade_closed", {
                "symbol": self.SYMBOL,
                "side": pos.side.upper(),
                "price": price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason
            })
        
        self.position = None
    
    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running = False
        
        if self.telegram.enabled:
            await self.telegram.send_message("🛑 *Julaba Bot Stopped*")
            await self.telegram.stop()
        
        if self.exchange:
            await self.exchange.close()
        
        logger.info(f"Final Balance: ${self.balance:,.2f} | Total P&L: ${self.stats.total_pnl:+,.2f}")


# ============== CLI Entry Point ==============

def main():
    parser = argparse.ArgumentParser(
        description="Julaba - AI-Enhanced Crypto Trading Bot"
    )
    parser.add_argument(
        "--paper-balance",
        type=float,
        default=None,
        help="Paper trading balance (enables paper mode)"
    )
    parser.add_argument(
        "--ai-confidence",
        type=float,
        default=0.7,
        help="AI confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--ai-mode",
        choices=["filter", "advisory", "autonomous", "hybrid"],
        default="autonomous",
        help="AI mode: filter (validate only), advisory (AI suggests), autonomous (AI trades), hybrid (AI scans + suggests)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="LINK/USDT",
        help="Trading symbol (e.g., LINK/USDT, BTC/USDT)"
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=60,
        help="AI proactive scan interval in seconds (default: 60 = 1 min)"
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=14400,
        help="Autonomous summary interval in seconds (default: 14400 = 4 hours)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run mode: log trades without executing"
    )
    parser.add_argument(
        "--daily-loss-limit",
        type=float,
        default=0.05,
        help="Daily loss limit as decimal (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--pretrain-ml",
        action="store_true",
        default=False,
        help="Pre-train ML model on historical data before starting"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Enable web dashboard at http://localhost:5000"
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=5000,
        help="Dashboard port (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Setup file logging before anything else
    setup_logging(args.log_level)
    
    bot = Julaba(
        paper_balance=args.paper_balance,
        ai_confidence=args.ai_confidence,
        ai_mode=args.ai_mode,
        log_level=args.log_level,
        symbol=args.symbol,
        scan_interval=args.scan_interval,
        summary_interval=args.summary_interval
    )
    
    # Apply CLI overrides
    bot.dry_run_mode = args.dry_run
    bot.daily_loss_limit = args.daily_loss_limit
    
    # Start dashboard if requested
    if args.dashboard:
        bot.dashboard_enabled = True
        bot.dashboard.port = args.dashboard_port
        bot.dashboard.start()
        logger.info(f"🖥️ Dashboard enabled at http://localhost:{args.dashboard_port}")
    
    # Pre-train ML model if requested
    if args.pretrain_ml:
        logger.info("🧠 Pre-training ML model on historical data...")
        asyncio.run(bot.pretrain_ml_model())
    
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
