#!/usr/bin/env python3
"""
Julaba - AI-Enhanced Crypto Trading Bot
Combines the original trading strategy with AI signal filtering and Telegram notifications.
"""

import asyncio
import argparse
import logging
import os
import sys
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
    detect_candlestick_patterns,
    calculate_drawdown_adjusted_risk,
    get_regime_analysis,
    ml_predict_regime,
    ml_record_trade,
    get_ml_classifier
)

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
    TP1_R = 1.0
    TP2_R = 2.0
    TP3_R = 3.0
    TP1_PCT = 0.4
    TP2_PCT = 0.3
    TP3_PCT = 0.3
    TRAIL_TRIGGER_R = 1.0
    TRAIL_OFFSET_R = 0.5
    WARMUP_BARS = 100  # Reduced from 200 for faster startup
    
    def __init__(
        self,
        paper_balance: Optional[float] = None,
        ai_confidence: float = 0.7,
        log_level: str = "INFO",
        ai_mode: str = "filter",  # "filter", "advisory", "autonomous", "hybrid"
        symbol: str = "LINK/USDT",
        scan_interval: int = 300
    ):
        # Set log level
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # Symbol (now configurable!)
        self.SYMBOL = symbol
        
        # AI Trading Mode
        # "filter" = AI only validates technical signals (default)
        # "advisory" = AI can suggest trades, requires Telegram confirmation
        # "autonomous" = AI can open trades directly with high confidence
        # "hybrid" = AI suggests via Telegram, you confirm
        self.ai_mode = ai_mode
        self.pending_ai_trade = None  # For advisory/hybrid mode confirmation
        self.last_ai_scan_time = None  # Rate limit AI scans
        self.ai_scan_interval = scan_interval  # Configurable scan interval
        
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
        """Get ML classifier stats for Telegram /ml command."""
        return get_ml_classifier().get_stats()

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
        
        ohlcv = await self.exchange.fetch_ohlcv(
            self.SYMBOL,
            self.BASE_TF,
            limit=self.WARMUP_BARS + 50
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
        
        try:
            while self.running:
                # Fetch latest candles
                ohlcv = await self.exchange.fetch_ohlcv(
                    self.SYMBOL,
                    self.BASE_TF,
                    limit=5
                )
                
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
                
                # Sleep before next iteration
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def _on_bar_close(self):
        """Handle a new aggregated bar close."""
        if len(self.bars_agg) < self.WARMUP_BARS:
            remaining = self.WARMUP_BARS - len(self.bars_agg)
            if remaining % 50 == 0:
                logger.info(f"Warming up... {remaining} bars remaining")
            return
        
        current_price = self.bars_agg.iloc[-1]["close"]
        atr = self._calculate_atr()
        
        # Cache values for Telegram commands
        self.cached_last_price = current_price
        self.cached_last_atr = atr
        
        if atr == 0:
            return
        
        # Check if trading is paused
        if self.paused:
            return
        
        # Generate signal from indicator
        df_signals = generate_signals(self.bars_agg.copy())
        signal = int(df_signals.iloc[-1].get("Side", 0))
        
        # Only process if no position and we have a signal
        if self.position is None and signal != 0:
            await self._process_signal(signal, current_price, atr)
        
        # AI Proactive Scan (if enabled and no position and no technical signal)
        if self.position is None and signal == 0 and self.ai_mode in ["advisory", "autonomous", "hybrid"]:
            await self._ai_proactive_scan(current_price, atr)
    
    async def _ai_proactive_scan(self, price: float, atr: float):
        """Let AI proactively scan for opportunities."""
        # Rate limit: only scan every 5 minutes
        now = datetime.now(timezone.utc)
        if self.last_ai_scan_time:
            elapsed = (now - self.last_ai_scan_time).total_seconds()
            if elapsed < self.ai_scan_interval:
                return
        
        self.last_ai_scan_time = now
        
        # AI scans the market
        opportunity = self.ai_filter.proactive_scan(
            df=self.bars_agg,
            current_price=price,
            atr=atr,
            symbol=self.SYMBOL
        )
        
        if not opportunity:
            return
        
        # Handle based on AI mode
        if self.ai_mode == "autonomous":
            # AI opens trade directly (requires 85%+ confidence from proactive_scan)
            logger.info(f"🤖 AI AUTONOMOUS: Opening {opportunity['action']} trade")
            await self._open_position(
                opportunity['signal'],
                price,
                atr,
                source="ai_autonomous"
            )
            
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

    async def _process_signal(self, signal: int, price: float, atr: float):
        """Process a trading signal through AI filter."""
        side = "LONG" if signal == 1 else "SHORT"
        
        # AI Analysis
        ai_result = self.ai_filter.analyze_signal(
            signal=signal,
            df=self.bars_agg,
            current_price=price,
            atr=atr,
            symbol=self.SYMBOL
        )
        
        # Record signal in history
        self.signal_history.append({
            "side": side,
            "price": price,
            "approved": ai_result["approved"],
            "confidence": ai_result["confidence"],
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
        size = risk_amount / risk_per_unit
        
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
        default="filter",
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
        default=300,
        help="AI proactive scan interval in seconds (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
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
        scan_interval=args.scan_interval
    )
    
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
