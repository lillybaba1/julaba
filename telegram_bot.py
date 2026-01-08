"""
Telegram Bot for Julaba Trading System
Provides real-time notifications and interactive commands.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

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
            self.bot = Bot(token=self.token)
            logger.info("Telegram bot initialized")
    
    async def start(self):
        """Start the Telegram bot with command handlers."""
        if not self.enabled:
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
/summary - Toggle summary notifications on/off

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
        """Handle /stop command."""
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
        
        # Get trading context for the AI
        context_info = ""
        if self.get_status:
            status = self.get_status()
            context_info += f"Bot Status: {'Connected' if status.get('connected') else 'Disconnected'}, "
            context_info += f"Mode: {status.get('mode', 'Unknown')}, "
            context_info += f"Balance: ${status.get('balance', 0):,.2f}\n"
        
        if self.get_positions:
            positions = self.get_positions()
            if positions:
                context_info += f"Open Position: {positions[0]['side']} {positions[0]['symbol']}\n"
            else:
                context_info += "No open positions\n"
        
        if self.get_market:
            market = self.get_market()
            if market.get('price', 0) > 0:
                context_info += f"Current Price: ${market.get('price', 0):,.4f}\n"
        
        # Check for trade execution requests
        msg_lower = user_message.lower()
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
            logger.info(f"⚡ Attempting to execute {side} trade via AI chat")
            side = "long" if should_execute_long else "short"
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
                await update.message.reply_text(response, parse_mode="Markdown")
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
