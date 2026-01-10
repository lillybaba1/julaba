"""
Telegram Bot for Julaba Trading System
Provides real-time notifications and interactive commands.
"""

import os
import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def sanitize_markdown(text: str) -> str:
    """Sanitize text for Telegram Markdown to prevent parsing errors."""
    if not text:
        return text
    
    # Escape special Markdown characters that might break parsing
    # But preserve intentional formatting like *bold* and _italic_
    
    # First, protect valid markdown patterns
    # Then escape problematic characters
    
    # Fix unbalanced asterisks and underscores
    # Count occurrences - if odd, escape the last one
    asterisk_count = text.count('*')
    if asterisk_count % 2 == 1:
        # Find last asterisk and escape it
        text = text[::-1].replace('*', '\\*', 1)[::-1]
    
    underscore_count = text.count('_')
    if underscore_count % 2 == 1:
        text = text[::-1].replace('_', '\\_', 1)[::-1]
    
    # Escape square brackets that aren't part of links
    # Simple approach: escape standalone brackets
    text = re.sub(r'\[(?![^\]]+\]\()', '\\[', text)
    text = re.sub(r'(?<!\])\](?!\()', '\\]', text)
    
    return text


# Telegram imports (optional - graceful fallback if not installed)
try:
    from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class TelegramNotifier:
    """
    Telegram bot for trading notifications and commands.
    """
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Check for valid credentials (not placeholders)
        token_valid = self.token and "your_" not in self.token.lower() and len(self.token) > 20
        chat_valid = self.chat_id and "your_" not in self.chat_id.lower() and self.chat_id.lstrip("-").isdigit()
        
        self.enabled = bool(token_valid and chat_valid and TELEGRAM_AVAILABLE)
        self.bot: Optional[Bot] = None
        self.app: Optional[Application] = None
        
        # Startup timestamp - ignore messages older than this
        self.startup_time: Optional[datetime] = None
        
        # Trading state reference (set by main bot)
        self.get_status: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None
        self.get_pnl: Optional[Callable] = None
        self.get_ai_stats: Optional[Callable] = None
        self.get_balance: Optional[Callable] = None
        self.get_trades: Optional[Callable] = None
        self.get_market: Optional[Callable] = None
        self.get_signals: Optional[Callable] = None
        self.do_stop: Optional[Callable] = None
        self.do_pause: Optional[Callable] = None
        self.do_resume: Optional[Callable] = None
        self.chat_with_ai: Optional[Callable] = None  # AI chat function
        # AI mode callbacks
        self.get_ai_mode: Optional[Callable] = None
        self.set_ai_mode: Optional[Callable] = None
        self.confirm_ai_trade: Optional[Callable] = None
        self.reject_ai_trade: Optional[Callable] = None
        self.execute_ai_trade: Optional[Callable] = None  # AI chat can execute trades
        self.close_ai_trade: Optional[Callable] = None  # AI chat can close positions
        self.get_intelligence: Optional[Callable] = None  # Intelligence summary
        self.get_ml_stats: Optional[Callable] = None  # ML classifier stats
        self.get_regime: Optional[Callable] = None  # Market regime analysis
        self.toggle_summary: Optional[Callable] = None  # Toggle summary notifications
        self.get_summary_status: Optional[Callable] = None  # Get summary status
        # NEW: Enhanced module callbacks
        self.get_risk_stats: Optional[Callable] = None  # Risk manager stats
        self.get_mtf_analysis: Optional[Callable] = None  # Multi-timeframe analysis
        self.run_backtest: Optional[Callable] = None  # Run backtest
        self.get_chart: Optional[Callable] = None  # Generate chart
        self.get_equity_curve: Optional[Callable] = None  # Equity curve data
        
        # Trading control state
        self.paused = False
        
        if not self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("Telegram bot disabled - package not installed")
            elif not token_valid:
                logger.info("Telegram bot disabled - valid TELEGRAM_BOT_TOKEN not set")
            elif not chat_valid:
                logger.info("Telegram bot disabled - valid TELEGRAM_CHAT_ID not set")
        else:
            # Defer Bot initialization to avoid Python 3.14 anyio import issues
            # Bot will be created in start() method instead
            self.bot = None
            logger.info("Telegram bot initialized (deferred)")
    
    async def start(self):
        """Start the Telegram bot with command handlers."""
        if not self.enabled:
            return
        
        # Create Bot instance here (deferred from __init__)
        if self.bot is None:
            try:
                self.bot = Bot(token=self.token)
            except Exception as e:
                logger.error(f"Failed to create Telegram Bot: {e}")
                self.enabled = False
                return
        
        self.app = Application.builder().token(self.token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self.app.add_handler(CommandHandler("ai", self._cmd_ai_stats))
        self.app.add_handler(CommandHandler("balance", self._cmd_balance))
        self.app.add_handler(CommandHandler("trades", self._cmd_trades))
        self.app.add_handler(CommandHandler("market", self._cmd_market))
        self.app.add_handler(CommandHandler("signals", self._cmd_signals))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("pause", self._cmd_pause))
        self.app.add_handler(CommandHandler("resume", self._cmd_resume))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        # AI mode commands
        self.app.add_handler(CommandHandler("aimode", self._cmd_aimode))
        self.app.add_handler(CommandHandler("confirm", self._cmd_confirm))
        self.app.add_handler(CommandHandler("reject", self._cmd_reject))
        # Intelligence commands
        self.app.add_handler(CommandHandler("intel", self._cmd_intel))
        self.app.add_handler(CommandHandler("ml", self._cmd_ml))
        self.app.add_handler(CommandHandler("regime", self._cmd_regime))
        self.app.add_handler(CommandHandler("summary", self._cmd_summary))
        # NEW: Enhanced commands
        self.app.add_handler(CommandHandler("risk", self._cmd_risk))
        self.app.add_handler(CommandHandler("mtf", self._cmd_mtf))
        self.app.add_handler(CommandHandler("backtest", self._cmd_backtest))
        self.app.add_handler(CommandHandler("chart", self._cmd_chart))
        
        # Add callback query handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))
        
        # Add message handler for normal chat (non-command messages)
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        
        # Add error handler for 409 conflicts
        async def error_handler(update, context):
            """Handle Telegram errors gracefully."""
            error = context.error
            if "409" in str(error) or "Conflict" in str(error):
                logger.warning("Telegram conflict detected (409) - will retry automatically")
            else:
                logger.error(f"Telegram error: {error}")
        
        self.app.add_error_handler(error_handler)
        
        # Record startup time to ignore old messages
        from datetime import datetime, timezone
        self.startup_time = datetime.now(timezone.utc)
        
        # Clear any pending updates before starting (prevents processing old /stop commands)
        try:
            # Fetch and discard any pending updates
            updates = await self.bot.get_updates(offset=-1, timeout=1)
            if updates:
                # Get the latest update_id and mark all previous as read
                latest_id = updates[-1].update_id
                await self.bot.get_updates(offset=latest_id + 1, timeout=1)
                logger.info(f"Cleared {len(updates)} pending Telegram updates")
        except Exception as e:
            logger.debug(f"Could not clear pending updates: {e}")
        
        # Start polling in background with infinite retries for conflicts
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"],
            bootstrap_retries=-1  # Retry indefinitely on startup conflicts
        )
        
        logger.info("Telegram bot started - listening for commands")
        await self.send_message("🤖 *Julaba Bot Started*\n\nType /help for commands")
    
    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
    
    async def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send a message to the configured chat."""
        if not self.enabled:
            return
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    # =========== Notification Methods ===========
    
    async def notify_signal(
        self,
        symbol: str,
        side: str,
        price: float,
        ai_approved: bool,
        confidence: float,
        reasoning: str
    ):
        """Notify about a new trading signal."""
        emoji = "🟢" if side == "LONG" else "🔴"
        status = "✅ APPROVED" if ai_approved else "❌ REJECTED"
        
        msg = f"""
{emoji} *New Signal: {side}*

📊 *Symbol:* `{symbol}`
💰 *Price:* `${price:,.4f}`
🤖 *AI Status:* {status}
📈 *Confidence:* `{confidence:.0%}`
💡 *Analysis:* {reasoning}
"""
        await self.send_message(msg)
    
    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float
    ):
        """Notify about a trade being opened."""
        emoji = "🟢" if side == "LONG" else "🔴"
        
        msg = f"""
{emoji} *TRADE OPENED*

📊 *{symbol}* - {side}
💰 *Entry:* `${entry_price:,.4f}`
📦 *Size:* `{size:.4f}`
🛑 *Stop Loss:* `${stop_loss:,.4f}`

🎯 *Take Profits:*
  TP1: `${tp1:,.4f}` (40%)
  TP2: `${tp2:,.4f}` (30%)
  TP3: `${tp3:,.4f}` (30%)
"""
        await self.send_message(msg)
    
    async def notify_ai_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        confidence: float,
        reasoning: str,
        mode: str
    ):
        """Notify about AI-initiated trade opportunity."""
        emoji = "🟢" if action == "LONG" else "🔴"
        
        if mode == "autonomous":
            # AI already opened the trade
            msg = f"""
🤖 *AI AUTONOMOUS TRADE*

{emoji} *{action}* on *{symbol}*
💰 *Price:* `${price:,.4f}`
📈 *Confidence:* `{confidence:.0%}`
💡 *Analysis:* {reasoning}

_Trade opened automatically by AI_
"""
            await self.send_message(msg)
        else:
            # Advisory/Hybrid - ask for confirmation
            msg = f"""
🤖 *AI TRADE SUGGESTION*

{emoji} *{action}* on *{symbol}*
💰 *Price:* `${price:,.4f}`
📈 *Confidence:* `{confidence:.0%}`
💡 *Analysis:* {reasoning}

⏳ *Awaiting your decision...*
Use /confirm or /reject, or tap below:
"""
            # Send with inline buttons
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Confirm Trade", callback_data="confirm_ai_trade"),
                    InlineKeyboardButton("❌ Reject", callback_data="reject_ai_trade")
                ]
            ])
            
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=msg,
                    parse_mode="Markdown",
                    reply_markup=keyboard
                )
            except Exception as e:
                logger.error(f"Failed to send AI trade notification: {e}")
    
    async def notify_tp_hit(
        self,
        symbol: str,
        tp_level: int,
        price: float,
        pnl: float,
        remaining_pct: float
    ):
        """Notify when a take profit level is hit."""
        msg = f"""
🎯 *TP{tp_level} HIT!*

📊 *{symbol}*
💰 *Exit Price:* `${price:,.4f}`
📈 *Realized P&L:* `${pnl:+,.2f}`
📦 *Remaining:* `{remaining_pct:.0%}`
"""
        await self.send_message(msg)
    
    async def notify_stop_loss(
        self,
        symbol: str,
        price: float,
        pnl: float
    ):
        """Notify when stop loss is hit."""
        msg = f"""
🛑 *STOP LOSS HIT*

📊 *{symbol}*
💰 *Exit Price:* `${price:,.4f}`
📉 *Loss:* `${pnl:,.2f}`
"""
        await self.send_message(msg)
    
    async def notify_trade_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        reason: str
    ):
        """Notify when a trade is fully closed."""
        emoji = "✅" if pnl >= 0 else "❌"
        
        msg = f"""
{emoji} *TRADE CLOSED*

📊 *{symbol}*
💰 *P&L:* `${pnl:+,.2f}` ({pnl_pct:+.2f}%)
📝 *Reason:* {reason}
"""
        await self.send_message(msg)
    
    async def notify_daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        losses: int,
        pnl: float,
        balance: float,
        win_rate: float
    ):
        """Send daily trading summary."""
        emoji = "📈" if pnl >= 0 else "📉"
        
        msg = f"""
📊 *Daily Summary - {date}*

{emoji} *Today's P&L:* `${pnl:+,.2f}`
💰 *Balance:* `${balance:,.2f}`

*Trades:* `{trades}` ({wins}W / {losses}L)
*Win Rate:* `{win_rate:.1f}%`

_Keep trading smart! 🤖_
"""
        await self.send_message(msg)
    
    # =========== Command Handlers ===========
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(
            "🤖 *Julaba Trading Bot*\n\n"
            "I'll send you real-time trading alerts and AI analysis.\n\n"
            "Use /help to see available commands.",
            parse_mode="Markdown"
        )
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        msg = """
🤖 *Julaba Commands*

📊 *Info Commands:*
/status - Bot status & connection info
/balance - Current balance
/positions - View open positions
/pnl - Show P&L summary
/stats - Detailed statistics

📈 *Trading Commands:*
/trades - Recent trade history
/signals - Recent signals detected
/market - Current market info
/ai - AI filter statistics
/chart - Price chart with levels

🤖 *AI Mode Commands:*
/aimode - View current AI mode
/aimode filter - AI validates signals only
/aimode advisory - AI suggests, you confirm
/aimode autonomous - AI trades directly
/aimode hybrid - AI scans + suggests
/confirm - Confirm pending AI trade
/reject - Reject pending AI trade

🧠 *Intelligence Commands:*
/intel - View intelligent trading features
/ml - Machine learning classifier stats
/regime - Current market regime analysis
/risk - Risk manager status & limits
/mtf - Multi-timeframe analysis

📉 *Analysis Commands:*
/backtest - Backtest strategy (7 days)
/backtest 30 - Backtest with custom days
/summary - Toggle summary notifications

⚙️ *Control Commands:*
/pause - Pause trading
/resume - Resume trading
/stop - Stop the bot

/help - Show this message
"""
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if self.get_status:
            s = self.get_status()
            
            # Determine status emoji
            if s.get('paused'):
                status_icon = "⏸️ PAUSED"
            elif s.get('has_position'):
                status_icon = "🟢 IN TRADE"
            else:
                status_icon = "🔵 WATCHING"
            
            # Calculate P&L percentage
            initial = s.get('initial_balance', 1)
            pnl_pct = ((s.get('balance', 0) - initial) / initial * 100) if initial > 0 else 0
            
            msg = f"""
📊 *Julaba Status*

*Connection*
🔌 Status: {'✅ Connected' if s.get('connected') else '❌ Disconnected'}
📈 Symbol: `{s.get('symbol', 'N/A')}`
⏱ Uptime: `{s.get('uptime', 'N/A')}`
🎮 Mode: `{s.get('mode', 'N/A')}`

*Trading*
{status_icon}
💵 Price: `${s.get('current_price', 0):,.4f}`
📏 ATR: `${s.get('atr', 0):,.4f}`
📍 Position: `{s.get('position_side', 'None')}`
{'💹 Unrealized: `$' + f"{s.get('position_pnl', 0):+,.2f}" + '`' if s.get('has_position') else ''}

*Performance*
💰 Balance: `${s.get('balance', 0):,.2f}`
📊 P&L: `${s.get('total_pnl', 0):+,.2f}` (`{pnl_pct:+.2f}%`)
🎯 Trades: `{s.get('total_trades', 0)}` ({s.get('wins', 0)}W / {s.get('losses', 0)}L)
📈 Win Rate: `{s.get('win_rate', 0):.1f}%`
🔍 Signals: `{s.get('signals_checked', 0)}` checked
"""
        else:
            msg = "⚠️ Status not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if self.get_positions:
            positions = self.get_positions()
            if positions:
                msg = "📦 *Open Positions*\n\n"
                for p in positions:
                    msg += f"• {p['symbol']}: {p['side']} @ ${p['entry']:.4f}\n"
                    msg += f"  Size: {p['size']:.4f} | P&L: ${p['pnl']:+.2f}\n\n"
            else:
                msg = "📦 *No open positions*"
        else:
            msg = "⚠️ Positions not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command."""
        if self.get_pnl:
            pnl = self.get_pnl()
            msg = f"""
💰 *P&L Summary*

📈 *Today:* `${pnl.get('today', 0):+,.2f}`
📊 *Total:* `${pnl.get('total', 0):+,.2f}`
🎯 *Win Rate:* `{pnl.get('win_rate', 0):.1%}`
📦 *Trades:* `{pnl.get('trades', 0)}`
"""
        else:
            msg = "⚠️ P&L not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def _cmd_ai_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ai command."""
        if self.get_ai_stats:
            stats = self.get_ai_stats()
            msg = f"""
🤖 *AI Filter Stats*

📊 *Total Signals:* `{stats.get('total_signals', 0)}`
✅ *Approved:* `{stats.get('approved', 0)}`
❌ *Rejected:* `{stats.get('rejected', 0)}`
📈 *Approval Rate:* `{stats.get('approval_rate', 'N/A')}`
"""
        else:
            msg = "⚠️ AI stats not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_intel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /intel command - show intelligent trading features."""
        if self.get_intelligence:
            i = self.get_intelligence()
            
            # Drawdown mode emoji
            mode = i.get('drawdown_mode', 'NORMAL')
            if mode == 'EMERGENCY':
                mode_emoji = "🚨"
            elif mode == 'CAUTIOUS':
                mode_emoji = "⚠️"
            elif mode == 'REDUCED':
                mode_emoji = "📉"
            else:
                mode_emoji = "✅"
            
            # Pattern info
            pattern_info = "None detected"
            pattern = i.get('pattern')
            if pattern and pattern.get('pattern'):
                p_dir = "🟢 Bullish" if pattern.get('bullish') else ("🔴 Bearish" if pattern.get('bullish') is False else "⚪ Neutral")
                pattern_info = f"{pattern.get('pattern')} ({p_dir})"
            
            # Regime info
            regime = i.get('regime', 'UNKNOWN')
            tradeable = "✅ Yes" if i.get('tradeable') else "❌ No"
            
            msg = f"""
🧠 *Intelligence Summary*

*Risk Management*
{mode_emoji} Mode: `{mode}`
📊 Drawdown: `{i.get('drawdown_pct', 0):.1f}%`
🔥 Win Streak: `{i.get('consecutive_wins', 0)}`
❄️ Loss Streak: `{i.get('consecutive_losses', 0)}`

*Market Analysis*
📈 Regime: `{regime}`
🎯 Tradeable: {tradeable}
📏 ADX: `{i.get('adx', 0):.1f}`
📐 Hurst: `{i.get('hurst', 0.5):.2f}`

*Pattern Detection*
🕯️ Pattern: `{pattern_info}`

*Machine Learning*
🤖 Status: `{i.get('ml_status', 'Not available')}`
"""
        else:
            msg = "⚠️ Intelligence data not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_ml(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ml command - show ML classifier stats."""
        if self.get_ml_stats:
            stats = self.get_ml_stats()
            
            trained = "✅ Yes" if stats.get('is_trained') else "❌ No"
            samples = stats.get('total_samples', 0)
            needed = stats.get('samples_until_training', 50)
            
            msg = f"""
🧠 *ML Classifier Stats*

*Training Status*
📚 Trained: {trained}
📊 Samples: `{samples}`
{'🎯 Samples until training: `' + str(needed) + '`' if needed > 0 else ''}

*Historical Performance*
"""
            if samples > 0:
                wins = stats.get('wins', 0)
                losses = stats.get('losses', 0)
                wr = stats.get('historical_win_rate', 0)
                msg += f"✅ Wins: `{wins}`\n"
                msg += f"❌ Losses: `{losses}`\n"
                msg += f"📈 Win Rate: `{wr:.1%}`\n"
            else:
                msg += "_No trades recorded yet_\n"
            
            # Top features if trained
            if stats.get('is_trained') and stats.get('top_features'):
                msg += "\n*Top Predictive Features*\n"
                for feat, imp in stats.get('top_features', [])[:3]:
                    msg += f"• `{feat}`: {imp:.2f}\n"
        else:
            msg = "⚠️ ML stats not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /regime command - show current market regime analysis."""
        if self.get_regime:
            r = self.get_regime()
            
            # Regime emoji
            regime_emoji = {
                'STRONG_TRENDING': '🚀',
                'TRENDING': '📈',
                'WEAK_TRENDING': '📊',
                'RANGING': '↔️',
                'CHOPPY': '🌊',
                'UNKNOWN': '❓'
            }.get(r.get('regime', 'UNKNOWN'), '❓')
            
            # Volatility emoji
            vol_emoji = {'high': '🔥', 'low': '❄️', 'normal': '✅'}.get(r.get('volatility', 'normal'), '✅')
            
            # Tradeable emoji
            trade_emoji = '✅' if r.get('tradeable') else '⚠️'
            
            msg = f"""
📊 *Market Regime Analysis*

{regime_emoji} *Regime:* `{r.get('regime', 'UNKNOWN')}`
{trade_emoji} *Tradeable:* {'Yes' if r.get('tradeable') else 'No'}

📈 *Indicators:*
• ADX (Trend Strength): `{r.get('adx', 0)}`
• Hurst Exponent: `{r.get('hurst', 0.5)}`
  _(>0.5 trending, <0.5 mean-reverting)_

{vol_emoji} *Volatility:*
• Level: `{r.get('volatility', 'normal').upper()}`
• Ratio: `{r.get('volatility_ratio', 1.0)}x`
"""
            
            # ML prediction if available
            if r.get('ml_prediction'):
                msg += f"""
🤖 *ML Prediction:*
• Predicted: `{r.get('ml_prediction')}`
• Confidence: `{r.get('ml_confidence', 0)}%`
"""
            
            msg += f"\n💡 _{r.get('description', '')}_"
        else:
            msg = "⚠️ Regime analysis not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summary command - toggle summary notifications on/off."""
        if self.toggle_summary and self.get_summary_status:
            # Toggle the state
            new_state = self.toggle_summary()
            
            if new_state:
                msg = """
✅ *Summary Notifications: ON*

📊 Periodic summaries will be sent automatically.
• Every few hours (configurable)
• Daily summary at 8:00 AM

Use /summary again to turn off.
"""
            else:
                msg = """
🔇 *Summary Notifications: OFF*

📊 Periodic summaries are now disabled.
You can still use:
• /status - Current status
• /pnl - Performance stats
• /intel - Intelligence overview

Use /summary again to turn on.
"""
        else:
            msg = "⚠️ Summary toggle not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    # ============== NEW ENHANCED COMMANDS ==============

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command - show risk manager status."""
        if self.get_risk_stats:
            r = self.get_risk_stats()
            
            can_trade_emoji = "✅" if r.get('can_trade') else "🛑"
            mode_emoji = {
                'NORMAL': '✅',
                'REDUCED': '⚠️',
                'CAUTIOUS': '🟡',
                'SEVERE': '🟠',
                'EMERGENCY': '🔴'
            }.get(r.get('dd_mode', 'NORMAL'), '❓')
            
            msg = f"""
🎯 *Risk Manager Status*

{can_trade_emoji} *Trading:* {'Allowed' if r.get('can_trade') else 'BLOCKED'}
{mode_emoji} *Mode:* `{r.get('dd_mode', 'NORMAL')}`
📊 *Reason:* {r.get('can_trade_reason', 'OK')}

*Position Sizing:*
├ Base Risk: `{r.get('base_risk', 2):.2%}`
├ Kelly Optimal: `{r.get('kelly_risk', 0.02):.2%}`
└ Adjusted Risk: `{r.get('adjusted_risk', 0.02):.2%}`

*Performance:*
├ Win Rate: `{r.get('win_rate', 0):.1%}`
├ Streak: {r.get('consecutive_wins', 0)}W / {r.get('consecutive_losses', 0)}L
└ Total Trades: `{r.get('total_trades', 0)}`

*Limits:*
├ Daily P&L: `${r.get('daily_pnl', 0):+.2f}`
├ Weekly P&L: `${r.get('weekly_pnl', 0):+.2f}`
├ Daily Limit: {'🛑 HIT' if r.get('daily_limit_hit') else '✅ OK'}
└ Weekly Limit: {'🛑 HIT' if r.get('weekly_limit_hit') else '✅ OK'}
"""
        else:
            msg = "⚠️ Risk manager not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_mtf(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mtf command - multi-timeframe analysis."""
        if self.get_mtf_analysis:
            r = self.get_mtf_analysis()
            
            if r.get('error'):
                msg = f"⚠️ {r['error']}"
            else:
                conf_emoji = "✅" if r.get('confirmed') else "❌"
                
                # Primary timeframe
                primary = r.get('primary', {})
                trend_3m = primary.get('trend', {}).get('direction', 'unknown')
                
                # Secondary timeframe
                secondary = r.get('secondary', {})
                trend_15m = secondary.get('trend', {}).get('direction', 'N/A') if secondary.get('trend') else 'N/A'
                
                # Higher timeframe
                higher = r.get('higher', {})
                trend_1h = higher.get('trend', {}).get('direction', 'N/A') if higher.get('trend') else 'N/A'
                
                msg = f"""
📊 *Multi-Timeframe Analysis*

{conf_emoji} *Confirmation:* `{r.get('recommendation', 'WAIT')}`
📈 *Confluence:* `{r.get('confluence_pct', 0)}%`
🎯 *Alignment Score:* `{r.get('alignment_score', 0):.2f}`

*Timeframe Trends:*
├ 3m: `{trend_3m.upper()}`
├ 15m: `{trend_15m.upper() if trend_15m != 'N/A' else 'N/A'}`
└ 1H: `{trend_1h.upper() if trend_1h != 'N/A' else 'N/A'}`

*Volume:*
└ Ratio: `{r.get('volume', {}).get('volume_ratio', 1.0):.2f}x` ({r.get('volume', {}).get('trend', 'normal')})

*Confirmations:* {len(r.get('confirmations', []))}
{chr(10).join('✅ ' + c for c in r.get('confirmations', [])[:3]) or '_None_'}

*Conflicts:* {len(r.get('conflicts', []))}
{chr(10).join('⚠️ ' + c for c in r.get('conflicts', [])[:3]) or '_None_'}

💡 _{r.get('message', '')}_
"""
        else:
            msg = "⚠️ MTF analysis not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backtest command - run historical backtest."""
        if not self.run_backtest:
            await update.message.reply_text("⚠️ Backtest not available")
            return
        
        # Parse days argument
        days = 7
        if context.args:
            try:
                days = int(context.args[0])
                days = min(max(days, 1), 90)  # Clamp to 1-90 days
            except ValueError:
                pass
        
        await update.message.reply_text(f"⏳ Running {days}-day backtest... This may take a moment.")
        
        try:
            result = await self.run_backtest(days)
            
            if result.get('error'):
                msg = f"❌ Backtest failed: {result['error']}"
            else:
                mc = result.get('monte_carlo', {})
                
                pnl_emoji = "📈" if result.get('total_pnl', 0) >= 0 else "📉"
                
                msg = f"""
📊 *Backtest Results ({days} days)*

{pnl_emoji} *Performance:*
├ Total P&L: `${result.get('total_pnl', 0):+,.2f}` ({result.get('total_pnl_pct', 0):+.1f}%)
├ Win Rate: `{result.get('win_rate', 0):.1f}%`
├ Profit Factor: `{result.get('profit_factor', 0):.2f}`
└ Sharpe Ratio: `{result.get('sharpe_ratio', 0):.2f}`

📈 *Trades:*
├ Total: `{result.get('total_trades', 0)}`
└ Max Drawdown: `{result.get('max_drawdown_pct', 0):.1f}%`

🎲 *Monte Carlo ({mc.get('simulations', 0)} sims):*
├ Median Final: `${mc.get('median_final_balance', 0):,.0f}`
├ 5th Percentile: `${mc.get('percentile_5', 0):,.0f}`
├ 95th Percentile: `${mc.get('percentile_95', 0):,.0f}`
├ Prob of Profit: `{mc.get('probability_profit', 0):.0f}%`
└ Worst-Case DD: `{mc.get('worst_case_drawdown', 0):.1f}%`
"""
        except Exception as e:
            msg = f"❌ Backtest error: {str(e)}"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chart command - generate and send price chart."""
        if not self.get_chart:
            await update.message.reply_text("⚠️ Chart generation not available")
            return
        
        await update.message.reply_text("📊 Generating chart...")
        
        try:
            chart_bytes = self.get_chart()
            
            if chart_bytes:
                from io import BytesIO
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=BytesIO(chart_bytes),
                    caption="📊 *Current Price Chart*\nBlue=Entry, Red=SL, Green=TP",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Could not generate chart (insufficient data or matplotlib not installed)")
        except Exception as e:
            logger.error(f"Chart command error: {e}")
            await update.message.reply_text(f"❌ Chart error: {str(e)}")

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        if self.get_balance:
            data = self.get_balance()
            change = data.get('change', 0)
            change_emoji = "📈" if change >= 0 else "📉"
            msg = f"""
💰 *Balance*

💵 *Current:* `${data.get('current', 0):,.2f}`
🏦 *Initial:* `${data.get('initial', 0):,.2f}`
{change_emoji} *Change:* `${change:+,.2f}` ({data.get('change_pct', 0):+.2f}%)
"""
        else:
            msg = "⚠️ Balance not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        if self.get_trades:
            trades = self.get_trades()
            if trades:
                msg = "📜 *Recent Trades*\n\n"
                for t in trades[-10:]:  # Last 10 trades
                    emoji = "✅" if t.get('pnl', 0) >= 0 else "❌"
                    msg += f"{emoji} {t.get('side', 'N/A')} @ ${t.get('entry', 0):.4f}\n"
                    msg += f"   P&L: `${t.get('pnl', 0):+.2f}` | {t.get('time', 'N/A')}\n\n"
            else:
                msg = "📜 *No trades yet*"
        else:
            msg = "⚠️ Trade history not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /market command."""
        if self.get_market:
            data = self.get_market()
            change = data.get('change_24h', 0)
            change_emoji = "🟢" if change >= 0 else "🔴"
            msg = f"""
📈 *Market Info*

📊 *Symbol:* `{data.get('symbol', 'N/A')}`
💰 *Price:* `${data.get('price', 0):,.4f}`
{change_emoji} *24h Change:* `{change:+.2f}%`
📊 *24h Volume:* `${data.get('volume_24h', 0):,.0f}`
📉 *24h Low:* `${data.get('low_24h', 0):,.4f}`
📈 *24h High:* `${data.get('high_24h', 0):,.4f}`
📏 *ATR:* `${data.get('atr', 0):,.4f}`
"""
        else:
            msg = "⚠️ Market data not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command."""
        if self.get_signals:
            signals = self.get_signals()
            if signals:
                msg = "📡 *Recent Signals*\n\n"
                for s in signals[-10:]:  # Last 10 signals
                    status = "✅" if s.get('approved') else "❌"
                    emoji = "🟢" if s.get('side') == 'LONG' else "🔴"
                    msg += f"{emoji} {s.get('side', 'N/A')} @ ${s.get('price', 0):.4f} {status}\n"
                    msg += f"   Confidence: `{s.get('confidence', 0):.0%}` | {s.get('time', 'N/A')}\n\n"
            else:
                msg = "📡 *No signals yet*"
        else:
            msg = "⚠️ Signal history not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        if self.get_pnl and self.get_ai_stats:
            pnl = self.get_pnl()
            ai = self.get_ai_stats()
            
            # Calculate additional stats
            total_trades = pnl.get('trades', 0)
            winning = pnl.get('winning', 0)
            losing = total_trades - winning
            
            msg = f"""
📊 *Detailed Statistics*

💰 *Performance:*
├ Today P&L: `${pnl.get('today', 0):+,.2f}`
├ Total P&L: `${pnl.get('total', 0):+,.2f}`
├ Win Rate: `{pnl.get('win_rate', 0):.1%}`
├ Winning: `{winning}` | Losing: `{losing}`
└ Total Trades: `{total_trades}`

🤖 *AI Filter:*
├ Signals Analyzed: `{ai.get('total_signals', 0)}`
├ Approved: `{ai.get('approved', 0)}`
├ Rejected: `{ai.get('rejected', 0)}`
└ Approval Rate: `{ai.get('approval_rate', 'N/A')}`

📈 *Strategy:*
├ Max Win: `${pnl.get('max_win', 0):+,.2f}`
├ Max Loss: `${pnl.get('max_loss', 0):,.2f}`
└ Avg Trade: `${pnl.get('avg_trade', 0):+,.2f}`
"""
        else:
            msg = "⚠️ Statistics not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - only process if message is recent (after bot startup)."""
        from datetime import datetime, timezone, timedelta
        
        # Check if message is from before bot startup (old queued message)
        msg_time = update.message.date
        if self.startup_time and msg_time:
            # Allow 10 second grace period for clock drift
            if msg_time < self.startup_time - timedelta(seconds=10):
                logger.info(f"Ignoring old /stop command from {msg_time} (startup was {self.startup_time})")
                return
        
        await update.message.reply_text(
            "⚠️ *Stopping bot...*\n\nThe bot will shut down gracefully.",
            parse_mode="Markdown"
        )
        if self.do_stop:
            await self.do_stop()

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        if self.do_pause:
            self.do_pause()
            self.paused = True
            await update.message.reply_text(
                "⏸ *Trading Paused*\n\nThe bot will not open new positions.\nExisting positions will still be managed.\n\nUse /resume to continue trading.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("⚠️ Pause not available", parse_mode="Markdown")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        if self.do_resume:
            self.do_resume()
            self.paused = False
            await update.message.reply_text(
                "▶️ *Trading Resumed*\n\nThe bot will now open new positions when signals are detected.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("⚠️ Resume not available", parse_mode="Markdown")

    async def _cmd_aimode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /aimode command - view or set AI trading mode."""
        if context.args and len(context.args) > 0:
            # Set mode
            new_mode = context.args[0].lower()
            if self.set_ai_mode and self.set_ai_mode(new_mode):
                mode_descriptions = {
                    "filter": "🔍 AI only validates technical signals",
                    "advisory": "💡 AI suggests trades, you confirm via Telegram",
                    "autonomous": "🤖 AI can open trades directly (85%+ confidence)",
                    "hybrid": "🔄 AI suggests trades when no technical signal"
                }
                await update.message.reply_text(
                    f"✅ *AI Mode Changed*\n\n"
                    f"Mode: `{new_mode}`\n"
                    f"{mode_descriptions.get(new_mode, '')}",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    "❌ Invalid mode. Use: `filter`, `advisory`, `autonomous`, or `hybrid`",
                    parse_mode="Markdown"
                )
        else:
            # Show current mode
            current_mode = self.get_ai_mode() if self.get_ai_mode else "filter"
            await update.message.reply_text(
                f"""🤖 *AI Trading Mode*

Current: `{current_mode}`

*Available Modes:*
• `filter` - AI only validates technical signals
• `advisory` - AI suggests trades, you confirm
• `autonomous` - AI opens trades directly (85%+)
• `hybrid` - AI suggests when no technical signal

Usage: `/aimode <mode>`
Example: `/aimode advisory`""",
                parse_mode="Markdown"
            )

    async def _cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /confirm command - confirm pending AI trade."""
        if self.confirm_ai_trade:
            result = await self.confirm_ai_trade()
            if result:
                await update.message.reply_text("✅ *AI Trade Confirmed*\n\nOpening position...", parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ No pending AI trade to confirm", parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ Confirm not available", parse_mode="Markdown")

    async def _cmd_reject(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reject command - reject pending AI trade."""
        if self.reject_ai_trade:
            result = await self.reject_ai_trade()
            if result:
                await update.message.reply_text("❌ *AI Trade Rejected*", parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ No pending AI trade to reject", parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ Reject not available", parse_mode="Markdown")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "confirm_ai_trade":
            if self.confirm_ai_trade:
                result = await self.confirm_ai_trade()
                if result:
                    await query.edit_message_text("✅ *AI Trade Confirmed* - Opening position...", parse_mode="Markdown")
                else:
                    await query.edit_message_text("⚠️ Trade expired or already handled", parse_mode="Markdown")
        
        elif query.data == "reject_ai_trade":
            if self.reject_ai_trade:
                await self.reject_ai_trade()
                await query.edit_message_text("❌ *AI Trade Rejected*", parse_mode="Markdown")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages - chat with AI."""
        user_message = update.message.text
        
        # Build AI context from SINGLE SOURCE OF TRUTH: full_state
        context_info = ""
        full_state = None
        
        # Get full system state - this is the ONLY source of context for AI
        if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
            try:
                full_state = self.get_full_system_state()
                
                # Parameters
                params = full_state.get('parameters', {})
                context_info += f"SYSTEM PARAMETERS:\n"
                context_info += f"  Risk: {params.get('risk_pct', 0.02)*100:.1f}% | ATR Mult: {params.get('atr_mult', 2.0)}\n"
                context_info += f"  TP Levels: {params.get('tp1_r', 1)}R/{params.get('tp2_r', 2)}R/{params.get('tp3_r', 3)}R\n"
                context_info += f"  AI Mode: {params.get('ai_mode', 'filter')} | AI Confidence: {params.get('ai_confidence', 0.7)*100:.0f}%\n"
                context_info += f"  Paused: {params.get('paused', False)}\n"
                context_info += f"  Daily Loss Limit: {params.get('daily_loss_limit', 0.05)*100:.1f}%"
                if params.get('daily_loss_triggered'):
                    context_info += " ⚠️ TRIGGERED"
                context_info += "\n"
                context_info += f"  Dry-Run Mode: {params.get('dry_run_mode', False)}\n\n"
                
                # Status (from full_state.status)
                status = full_state.get('status', {})
                context_info += f"BOT STATUS:\n"
                context_info += f"  Connected: {'Yes' if status.get('connected') else 'No'}\n"
                context_info += f"  Mode: {status.get('mode', 'Unknown')}\n"
                context_info += f"  Uptime: {status.get('uptime', 'N/A')}\n"
                context_info += f"  Balance: ${status.get('balance', 0):,.2f} (Initial: ${status.get('initial_balance', 0):,.2f})\n"
                context_info += f"  Total Trades: {status.get('total_trades', 0)} | Win Rate: {status.get('win_rate', 0):.1f}%\n\n"
                
                # Position (from full_state.position)
                positions = full_state.get('position', [])
                if positions and len(positions) > 0:
                    pos = positions[0]
                    context_info += f"OPEN POSITION:\n"
                    context_info += f"  {pos.get('side', 'N/A')} {pos.get('symbol', 'N/A')}\n"
                    context_info += f"  Entry: ${pos.get('entry', 0):,.4f} | Size: {pos.get('size', 0):.4f}\n"
                    context_info += f"  Unrealized P&L: ${pos.get('pnl', 0):,.2f}\n\n"
                else:
                    context_info += "POSITION: None\n\n"
                
                # P&L (from full_state.pnl)
                pnl = full_state.get('pnl', {})
                context_info += f"P&L:\n"
                context_info += f"  Today: ${pnl.get('today', 0):,.2f}\n"
                context_info += f"  Total: ${pnl.get('total', 0):,.2f}\n"
                context_info += f"  Win Rate: {pnl.get('win_rate', 0)*100:.1f}%\n"
                context_info += f"  Total Trades: {pnl.get('trades', 0)} ({pnl.get('winning', 0)} wins)\n\n"
                
                # Market (from full_state.market)
                market = full_state.get('market', {})
                if market.get('price', 0) > 0:
                    context_info += f"MARKET:\n"
                    context_info += f"  {market.get('symbol', 'N/A')}: ${market.get('price', 0):,.4f}\n"
                    context_info += f"  24h Change: {market.get('change_24h', 0):.2f}%\n"
                    context_info += f"  ATR: ${market.get('atr', 0):,.4f}\n\n"
                
                # ML Model (from full_state.ml)
                ml = full_state.get('ml', {})
                samples = ml.get('total_samples', 0)
                samples_until = ml.get('samples_until_training', 50)
                is_trained = ml.get('is_trained', False)
                
                context_info += f"ML MODEL:\n"
                if is_trained:
                    win_rate = ml.get('historical_win_rate', 0)
                    context_info += f"  Status: Trained ✅\n"
                    context_info += f"  Samples: {samples}\n"
                    context_info += f"  Historical Win Rate: {win_rate*100:.0f}%\n"
                else:
                    context_info += f"  Status: Learning\n"
                    context_info += f"  Progress: {samples}/{samples + samples_until} samples ({samples_until} more needed)\n"
                context_info += "\n"
                
                # Regime
                context_info += f"MARKET REGIME: {full_state.get('regime', 'unknown')}\n\n"
                
                # Market Scan - Top pairs by volatility
                market_scan = full_state.get('market_scan', {})
                if market_scan.get('top_pairs'):
                    context_info += f"MARKET SCANNER (top by score):\n"
                    context_info += f"  Currently trading: {market_scan.get('current_symbol', 'N/A')}\n"
                    for p in market_scan.get('top_pairs', []):
                        context_info += f"  • {p['symbol']}: ${p['price']:,.2f} | Score:{p.get('score',0):.0f} | RSI:{p.get('rsi',50):.0f} | {p.get('trend','n/a')}\n"
                    context_info += "\n"
                
                # Recent signals
                signals = full_state.get('signals', [])
                if signals:
                    context_info += f"RECENT SIGNALS ({len(signals)}):\n"
                    for sig in signals[-3:]:
                        context_info += f"  • {sig.get('direction', 'N/A')} @ {sig.get('time', 'N/A')}\n"
                    context_info += "\n"
                    
            except Exception as e:
                logger.warning(f"Could not get full system state: {e}")
                # Fallback to individual getters only if full_state fails
                if self.get_status:
                    status = self.get_status()
                    context_info += f"Bot Status: {'Connected' if status.get('connected') else 'Disconnected'}, "
                    context_info += f"Balance: ${status.get('balance', 0):,.2f}\n"
        
        # Check for PARAMETER CHANGE commands (set risk 3%, change mode to autonomous, etc)
        msg_lower = user_message.lower()
        param_change_result = None
        
        logger.info(f"🔧 Checking for param changes in: '{msg_lower}'")
        
        if hasattr(self, 'set_system_param') and self.set_system_param:
            import re
            # Match patterns like "set risk to 3%", "change ai confidence to 80%", "set tp1 to 1.5"
            set_patterns = [
                (r'set\s+risk\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%?', 'risk_pct', lambda x: float(x)/100),
                (r'(?:set|change)\s+(?:ai\s+)?confidence\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%?', 'ai_confidence', lambda x: float(x)/100),
                (r'(?:set|change)\s+(?:ai\s+)?mode\s+(?:to\s+)?(\w+)', 'ai_mode', str),
                (r'set\s+atr\s+(?:mult(?:iplier)?\s+)?(?:to\s+)?(\d+(?:\.\d+)?)', 'atr_mult', float),
                (r'set\s+tp1\s+(?:to\s+)?(\d+(?:\.\d+)?)', 'tp1_r', float),
                (r'set\s+tp2\s+(?:to\s+)?(\d+(?:\.\d+)?)', 'tp2_r', float),
                (r'set\s+tp3\s+(?:to\s+)?(\d+(?:\.\d+)?)', 'tp3_r', float),
                (r'(?:pause|stop)\s+(?:the\s+)?(?:bot|trading)', 'paused', lambda x: True),
                (r'(?:resume|start|unpause)\s+(?:the\s+)?(?:bot|trading)', 'paused', lambda x: False),
                # More flexible patterns for mode changes - these catch various phrasings
                (r'\bautonomous\b', 'ai_mode', lambda x: 'autonomous'),
                (r'\bfilter\b(?!\s+mode)', 'ai_mode', lambda x: 'filter'),
                (r'\bhybrid\b', 'ai_mode', lambda x: 'hybrid'),
                (r'\badvisory\b', 'ai_mode', lambda x: 'advisory'),
                (r'switch\s+(?:to\s+)?(\w+)', 'ai_mode', str),
                (r'mode\s*[=:]\s*(\w+)', 'ai_mode', str),
                (r'go\s+(?:to\s+)?(\w+)\s+mode', 'ai_mode', str),
                (r'enable\s+(\w+)\s+mode', 'ai_mode', str),
                (r'use\s+(\w+)\s+mode', 'ai_mode', str),
            ]
            
            for pattern, param, converter in set_patterns:
                match = re.search(pattern, msg_lower)
                if match:
                    try:
                        # Handle patterns with and without capture groups
                        if match.groups() and match.group(1):
                            value = converter(match.group(1))
                        else:
                            value = converter(True)
                        logger.info(f"🔧 Matched pattern '{pattern}' → {param}={value}")
                        param_change_result = self.set_system_param(param, value)
                        if param_change_result.get('success'):
                            context_info += f"\n\n✅ PARAMETER CHANGED: {param_change_result.get('message')}"
                            logger.info(f"✅ Parameter change success: {param_change_result.get('message')}")
                        else:
                            context_info += f"\n\n❌ PARAMETER CHANGE FAILED: {param_change_result.get('message')}"
                            logger.warning(f"❌ Parameter change failed: {param_change_result.get('message')}")
                        break
                    except Exception as e:
                        logger.error(f"Parameter change error: {e}")
        else:
            logger.warning("🔧 set_system_param not available!")
        
        # Check for trade execution requests
        trade_keywords_long = ["buy", "go long", "open long", "enter long", "long now", "buy now", 
                               "execute long", "execute buy", "yes buy", "yes long", "do it", "execute",
                               "execute trade", "make the trade", "open the trade", "enter the trade"]
        trade_keywords_short = ["sell", "go short", "open short", "enter short", "short now", "sell now",
                                "execute short", "execute sell", "yes short", "yes sell"]
        close_keywords = ["close", "exit", "close position", "exit position", "close trade", "take profit",
                          "close it", "exit trade", "close now", "exit now"]
        
        should_execute_long = any(kw in msg_lower for kw in trade_keywords_long)
        should_execute_short = any(kw in msg_lower for kw in trade_keywords_short)
        should_close = any(kw in msg_lower for kw in close_keywords) and not (should_execute_long or should_execute_short)
        
        logger.info(f"🔍 Keyword check: msg='{user_message}' long={should_execute_long} short={should_execute_short} close={should_close}")
        
        # Execute trade if requested
        if (should_execute_long or should_execute_short) and self.execute_ai_trade:
            side = "long" if should_execute_long else "short"
            logger.info(f"⚡ Attempting to execute {side} trade via AI chat")
            result = await self.execute_ai_trade(side)
            
            logger.info(f"💼 Trade result: {result}")
            
            if result["success"]:
                # Add trade info to context for AI response
                context_info += f"\n\n✅ TRADE JUST EXECUTED: {result['message']}"
            else:
                context_info += f"\n\n❌ TRADE FAILED: {result['message']}"
        
        # Close position if requested
        elif should_close and self.close_ai_trade:
            logger.info(f"🔴 Attempting to close position via AI chat")
            result = await self.close_ai_trade()
            
            logger.info(f"💼 Close result: {result}")
            
            if result["success"]:
                context_info += f"\n\n✅ POSITION CLOSED: {result['message']}"
            else:
                context_info += f"\n\n❌ CLOSE FAILED: {result['message']}"
        
        # Use AI to generate response
        if self.chat_with_ai:
            try:
                response = await self.chat_with_ai(user_message, context_info)
                
                # === EXECUTE AI COMMAND BLOCKS ===
                # Parse and execute any ```command blocks in AI response
                import re
                command_pattern = r'```command\s*\n?\s*(\{[^}]+\})\s*\n?```'
                command_matches = re.findall(command_pattern, response, re.IGNORECASE | re.DOTALL)
                
                command_results = []
                for cmd_json in command_matches:
                    try:
                        import json
                        cmd = json.loads(cmd_json.strip())
                        action = cmd.get('action', '')
                        
                        logger.info(f"🤖 AI COMMAND: {cmd}")
                        
                        if action == 'set_param' and self.set_system_param:
                            param = cmd.get('param', '')
                            value = cmd.get('value')
                            result = self.set_system_param(param, value)
                            if result.get('success'):
                                command_results.append(f"✅ {result.get('message')}")
                                logger.info(f"✅ AI set {param} = {value}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                                logger.warning(f"❌ AI failed to set {param}: {result.get('message')}")
                        
                        elif action == 'open_trade' and self.execute_ai_trade:
                            side = cmd.get('side', 'long')
                            result = await self.execute_ai_trade(side)
                            if result.get('success'):
                                command_results.append(f"✅ {result.get('message')}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                        
                        elif action == 'close_trade' and self.close_ai_trade:
                            result = await self.close_ai_trade()
                            if result.get('success'):
                                command_results.append(f"✅ {result.get('message')}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                        
                    except json.JSONDecodeError as je:
                        logger.error(f"AI command JSON error: {je}")
                    except Exception as e:
                        logger.error(f"AI command execution error: {e}")
                
                # Remove command blocks from displayed response
                clean_response = re.sub(command_pattern, '', response, flags=re.IGNORECASE | re.DOTALL).strip()
                
                # Append command results if any
                if command_results:
                    clean_response += "\n\n🔧 *Actions Executed:*\n" + "\n".join(command_results)
                
                # Sanitize markdown and send
                clean_response = sanitize_markdown(clean_response)
                
                # Try sending with Markdown, fall back to plain text if parsing fails
                try:
                    await update.message.reply_text(clean_response, parse_mode="Markdown")
                except Exception as markdown_error:
                    logger.warning(f"Markdown parsing failed, sending as plain text: {markdown_error}")
                    await update.message.reply_text(clean_response)
            except Exception as e:
                logger.error(f"AI chat error: {e}")
                await update.message.reply_text(
                    "🤖 Sorry, I had trouble processing that. Try asking again or use /help to see available commands.",
                    parse_mode="Markdown"
                )
        else:
            # Fallback if AI not available
            await update.message.reply_text(
                "🤖 Hi! I'm Julaba, your trading assistant.\n\n"
                "Use /help to see what I can do, or ask me anything about trading!",
                parse_mode="Markdown"
            )


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier() -> TelegramNotifier:
    """Get or create the Telegram notifier singleton."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
