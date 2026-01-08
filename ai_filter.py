"""
AI Signal Filter Module
Validates trading signals using AI analysis before execution.
Uses Google Gemini for cost-effective AI analysis.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

# Persistent history file paths
HISTORY_DIR = Path(__file__).parent
AI_HISTORY_FILE = HISTORY_DIR / "ai_history.json"
TRADE_HISTORY_FILE = HISTORY_DIR / "trade_history.json"
CHAT_HISTORY_FILE = HISTORY_DIR / "chat_history.json"


class AISignalFilter:
    """
    AI-powered signal filter that analyzes market conditions
    and validates trading signals before execution.
    Uses Google Gemini (free tier available).
    
    STRICT MODE: Higher threshold, skeptic prompt, loss cooldown.
    """
    
    def __init__(self, confidence_threshold: float = 0.80):
        """
        Initialize the AI Signal Filter.
        
        Args:
            confidence_threshold: Minimum confidence (0-1) required to approve a trade
        """
        self.confidence_threshold = confidence_threshold
        self.loss_cooldown_threshold = 0.90  # Require 90% after a loss
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.use_ai = bool(self.api_key and "your_" not in self.api_key.lower())
        self.trade_history = []
        self.model = None
        
        # Trading performance tracking - load from persistent storage
        self.recent_trades = []  # List of {"result": "win"/"loss", "pnl": float, "time": str}
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_wins = 0
        self.total_losses = 0
        
        # Chat history for context
        self.chat_history: List[Dict[str, str]] = []
        self.max_chat_history = 20  # Keep last 20 messages for context
        
        # Load persistent history
        self._load_persistent_history()
        
        if self.use_ai:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini 2.5 Flash AI filter initialized (fast mode)")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.use_ai = False
        
        if not self.use_ai:
            logger.info("GEMINI_API_KEY not set - AI filter will use rule-based analysis")
    
    def record_trade_result(self, is_win: bool, pnl: float):
        """Record a trade result to inform future AI decisions."""
        result = "win" if is_win else "loss"
        self.recent_trades.append({
            "result": result,
            "pnl": pnl,
            "time": datetime.utcnow().strftime("%H:%M:%S")
        })
        # Keep only last 10 trades
        if len(self.recent_trades) > 10:
            self.recent_trades = self.recent_trades[-10:]
        
        if is_win:
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        logger.info(f"Trade recorded: {result} ${pnl:+.2f} | Streak: {self.consecutive_wins}W / {self.consecutive_losses}L")
        
        # Save to persistent storage
        self._save_persistent_history()
    
    def _load_persistent_history(self):
        """Load trading history and chat history from disk."""
        # Load trade history
        if TRADE_HISTORY_FILE.exists():
            try:
                with open(TRADE_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.recent_trades = data.get("recent_trades", [])
                    self.total_wins = data.get("total_wins", 0)
                    self.total_losses = data.get("total_losses", 0)
                    self.consecutive_wins = data.get("consecutive_wins", 0)
                    self.consecutive_losses = data.get("consecutive_losses", 0)
                    logger.info(f"Loaded trade history: {self.total_wins}W/{self.total_losses}L")
            except Exception as e:
                logger.warning(f"Failed to load trade history: {e}")
        
        # Load chat history
        if CHAT_HISTORY_FILE.exists():
            try:
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.chat_history = data.get("messages", [])[-self.max_chat_history:]
                    logger.info(f"Loaded {len(self.chat_history)} chat messages from history")
            except Exception as e:
                logger.warning(f"Failed to load chat history: {e}")
    
    def _save_persistent_history(self):
        """Save trading history and chat history to disk."""
        # Save trade history
        try:
            trade_data = {
                "recent_trades": self.recent_trades[-10:],
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "consecutive_wins": self.consecutive_wins,
                "consecutive_losses": self.consecutive_losses,
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(TRADE_HISTORY_FILE, 'w') as f:
                json.dump(trade_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save trade history: {e}")
    
    def _save_chat_history(self):
        """Save chat history to disk."""
        try:
            chat_data = {
                "messages": self.chat_history[-self.max_chat_history:],
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump(chat_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")
    
    def _get_performance_context(self) -> Dict[str, Any]:
        """Get recent trading performance for AI context."""
        total = self.total_wins + self.total_losses
        win_rate = (self.total_wins / total * 100) if total > 0 else 0
        
        recent_pnl = sum(t["pnl"] for t in self.recent_trades[-5:]) if self.recent_trades else 0
        
        return {
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "recent_pnl": round(recent_pnl, 2),
            "last_trade_was_loss": self.consecutive_losses > 0
        }
    
    def _get_market_hours_context(self) -> Dict[str, Any]:
        """Get market timing context."""
        now = datetime.utcnow()
        hour = now.hour
        
        # Crypto market sessions (approximate)
        if 0 <= hour < 8:
            session = "Asia"
            activity = "moderate"
        elif 8 <= hour < 14:
            session = "Europe"
            activity = "high"
        elif 14 <= hour < 21:
            session = "US"
            activity = "high"
        else:
            session = "Late US/Early Asia"
            activity = "low"
        
        # Weekend check (lower liquidity)
        is_weekend = now.weekday() >= 5
        
        return {
            "session": session,
            "activity_level": activity,
            "is_weekend": is_weekend,
            "hour_utc": hour
        }
    
    def analyze_signal(
        self,
        signal: int,  # 1 = long, -1 = short, 0 = none
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Analyze a trading signal and return AI validation result.
        
        Returns:
            Dict with keys: approved, confidence, reasoning, risk_assessment
        """
        if signal == 0:
            return {
                "approved": False,
                "confidence": 0.0,
                "reasoning": "No signal to analyze",
                "risk_assessment": "N/A"
            }
        
        # Gather market context
        context = self._build_market_context(df, current_price, atr)
        
        if self.use_ai:
            result = self._ai_analysis(signal, context, symbol)
        else:
            result = self._rule_based_analysis(signal, context, symbol)
        
        # Log the analysis
        self._log_analysis(signal, symbol, result)
        
        return result
    
    def _build_market_context(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """Build market context from DataFrame."""
        
        recent = df.tail(20)
        
        # Calculate key metrics
        price_change_1h = (current_price - recent.iloc[-12]["close"]) / recent.iloc[-12]["close"] * 100 if len(recent) >= 12 else 0
        price_change_5m = (current_price - recent.iloc[-1]["close"]) / recent.iloc[-1]["close"] * 100
        
        # Volume analysis
        avg_volume = recent["volume"].mean()
        current_volume = recent.iloc[-1]["volume"]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Trend analysis
        sma_10 = recent["close"].tail(10).mean()
        sma_20 = recent["close"].tail(20).mean()
        trend = "bullish" if sma_10 > sma_20 else "bearish"
        
        # Volatility
        volatility = atr / current_price * 100  # ATR as % of price
        
        return {
            "current_price": current_price,
            "price_change_1h": round(price_change_1h, 2),
            "price_change_5m": round(price_change_5m, 3),
            "volume_ratio": round(volume_ratio, 2),
            "trend": trend,
            "volatility_pct": round(volatility, 2),
            "atr": round(atr, 4),
            "sma_10": round(sma_10, 4),
            "sma_20": round(sma_20, 4)
        }
    
    def _ai_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Use Google Gemini API for signal analysis with SKEPTIC MODE."""
        try:
            signal_type = "LONG" if signal == 1 else "SHORT"
            
            # Get additional context
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Determine if we need extra caution
            extra_caution = ""
            if perf["last_trade_was_loss"]:
                extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
            if perf["consecutive_losses"] >= 2:
                extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve."
            if market["is_weekend"]:
                extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk of false moves."
            if market["activity_level"] == "low":
                extra_caution += "\n🌙 LOW ACTIVITY HOURS: Increased slippage risk."
            
            prompt = f"""You are a SKEPTICAL crypto trading supervisor. Your job is to PROTECT capital by rejecting bad trades.

=== SIGNAL ===
Proposed Trade: {signal_type} on {symbol}

=== MARKET DATA ===
Current Price: ${context['current_price']}
1-Hour Price Change: {context['price_change_1h']}%
Volume Ratio (vs avg): {context['volume_ratio']}x
Trend (SMA10 vs SMA20): {context['trend']}
Volatility (ATR%): {context['volatility_pct']}%

=== TRADING PERFORMANCE ===
Total Trades: {perf['total_trades']}
Win Rate: {perf['win_rate']}%
Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L
Recent P&L (last 5): ${perf['recent_pnl']}

=== MARKET SESSION ===
Session: {market['session']} ({market['hour_utc']}:00 UTC)
Activity Level: {market['activity_level']}
Weekend: {market['is_weekend']}
{extra_caution}

=== YOUR TASK ===
FIRST, list 3 reasons why this trade could FAIL.
THEN, decide if the setup is strong enough to overcome those risks.

Only approve if you are genuinely confident. When in doubt, REJECT.

Respond ONLY with this JSON format, no other text:
{{"reasons_against": ["reason1", "reason2", "reason3"], "approved": false, "confidence": 0.65, "reasoning": "why approved or rejected", "risk_assessment": "low/medium/high"}}"""

            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            # Clean up common issues
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            # Ensure required fields exist
            result.setdefault("approved", False)
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "AI analysis")
            result.setdefault("risk_assessment", "medium")
            result.setdefault("reasons_against", [])
            
            # Apply confidence threshold with LOSS COOLDOWN
            perf = self._get_performance_context()
            
            if perf["consecutive_losses"] >= 2:
                # After 2+ consecutive losses, require 90% confidence
                required_threshold = self.loss_cooldown_threshold
                logger.info(f"Loss cooldown active: requiring {required_threshold:.0%} confidence")
            elif perf["last_trade_was_loss"]:
                # After 1 loss, require 85% confidence
                required_threshold = 0.85
            else:
                required_threshold = self.confidence_threshold
            
            result["approved"] = result["approved"] and result["confidence"] >= required_threshold
            result["threshold_used"] = required_threshold
            
            # Log the skeptic analysis
            if result.get("reasons_against"):
                logger.info(f"AI reasons against trade: {result['reasons_against']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}, falling back to rules")
            return self._rule_based_analysis(signal, context, symbol)
    
    def _rule_based_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Rule-based signal validation when AI is unavailable."""
        
        confidence = 0.5  # Start neutral
        reasons = []
        risk = "medium"
        
        signal_type = "LONG" if signal == 1 else "SHORT"
        
        # Rule 1: Trend alignment
        if (signal == 1 and context["trend"] == "bullish") or \
           (signal == -1 and context["trend"] == "bearish"):
            confidence += 0.15
            reasons.append(f"Signal aligns with {context['trend']} trend")
        else:
            confidence -= 0.1
            reasons.append(f"Counter-trend trade ({context['trend']} market)")
        
        # Rule 2: Volume confirmation
        if context["volume_ratio"] > 1.2:
            confidence += 0.1
            reasons.append("Strong volume confirmation")
        elif context["volume_ratio"] < 0.5:
            confidence -= 0.1
            reasons.append("Low volume - weak confirmation")
        
        # Rule 3: Volatility check
        if context["volatility_pct"] > 3:
            confidence -= 0.1
            risk = "high"
            reasons.append("High volatility environment")
        elif context["volatility_pct"] < 1:
            confidence += 0.05
            risk = "low"
            reasons.append("Low volatility - stable conditions")
        
        # Rule 4: Recent momentum
        if (signal == 1 and context["price_change_1h"] > 0) or \
           (signal == -1 and context["price_change_1h"] < 0):
            confidence += 0.1
            reasons.append("Momentum supports direction")
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "approved": confidence >= self.confidence_threshold,
            "confidence": round(confidence, 2),
            "reasoning": "; ".join(reasons),
            "risk_assessment": risk
        }
    
    def _log_analysis(
        self,
        signal: int,
        symbol: str,
        result: Dict[str, Any]
    ):
        """Log analysis for tracking."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "signal": "LONG" if signal == 1 else "SHORT",
            **result
        }
        self.trade_history.append(entry)
        
        status = "✅ APPROVED" if result["approved"] else "❌ REJECTED"
        logger.info(
            f"AI Filter {status}: {symbol} {entry['signal']} | "
            f"Confidence: {result['confidence']:.0%} | "
            f"Risk: {result['risk_assessment']} | "
            f"Reason: {result['reasoning']}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        if not self.trade_history:
            return {"total_signals": 0, "approved": 0, "rejected": 0}
        
        approved = sum(1 for t in self.trade_history if t["approved"])
        return {
            "total_signals": len(self.trade_history),
            "approved": approved,
            "rejected": len(self.trade_history) - approved,
            "approval_rate": f"{approved / len(self.trade_history):.1%}"
        }
    
    def proactive_scan(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        AI proactively scans market for opportunities (without technical signal).
        Returns a trade suggestion or None.
        """
        if not self.use_ai:
            return None
        
        try:
            context = self._build_market_context(df, current_price, atr)
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            prompt = f"""You are an expert crypto trader with full autonomy to identify trading opportunities.

=== MARKET DATA for {symbol} ===
Current Price: ${context['current_price']}
1-Hour Price Change: {context['price_change_1h']}%
5-Min Price Change: {context['price_change_5m']}%
Volume Ratio (vs avg): {context['volume_ratio']}x
Trend (SMA10 vs SMA20): {context['trend']}
Volatility (ATR%): {context['volatility_pct']}%
SMA10: ${context['sma_10']}
SMA20: ${context['sma_20']}

=== TRADING PERFORMANCE ===
Total Trades: {perf['total_trades']}
Win Rate: {perf['win_rate']}%
Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L
Recent P&L (last 5): ${perf['recent_pnl']}

=== MARKET SESSION ===
Session: {market['session']} ({market['hour_utc']}:00 UTC)
Activity Level: {market['activity_level']}
Weekend: {market['is_weekend']}

=== YOUR TASK ===
Analyze the market RIGHT NOW. Is there a clear trading opportunity?

ONLY suggest a trade if you see:
- Strong trend with momentum
- Good volume confirmation
- Favorable risk/reward
- No major red flags

If no clear opportunity, respond with: {{"action": "wait", "reasoning": "why waiting"}}

If you see an opportunity, respond with:
{{"action": "LONG" or "SHORT", "confidence": 0.0-1.0, "reasoning": "why this trade", "risk_assessment": "low/medium/high", "suggested_risk_pct": 0.01-0.03}}

Respond ONLY with valid JSON, no other text:"""

            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            
            if result.get("action") == "wait":
                logger.debug(f"AI proactive scan: No opportunity - {result.get('reasoning', 'N/A')}")
                return None
            
            if result.get("action") in ["LONG", "SHORT"]:
                confidence = result.get("confidence", 0.5)
                
                # Require 85% confidence for AI-initiated trades
                if confidence >= 0.85:
                    logger.info(
                        f"🤖 AI OPPORTUNITY: {result['action']} {symbol} | "
                        f"Confidence: {confidence:.0%} | {result.get('reasoning', '')}"
                    )
                    return {
                        "action": result["action"],
                        "signal": 1 if result["action"] == "LONG" else -1,
                        "confidence": confidence,
                        "reasoning": result.get("reasoning", "AI identified opportunity"),
                        "risk_assessment": result.get("risk_assessment", "medium"),
                        "suggested_risk_pct": result.get("suggested_risk_pct", 0.02),
                        "source": "ai_proactive"
                    }
                else:
                    logger.debug(f"AI saw potential but confidence too low: {confidence:.0%}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"AI proactive scan error: {e}")
            return None

    async def chat(self, user_message: str, trading_context: str = "") -> str:
        """
        Chat with the AI about trading, market analysis, or general questions.
        Maintains conversation history for context across messages.
        
        Args:
            user_message: The user's message
            trading_context: Current trading context (positions, balance, etc.)
            
        Returns:
            AI response string
        """
        if not self.use_ai or not self.model:
            return self._simple_chat_response(user_message)
        
        try:
            # Build conversation history context
            history_text = ""
            if self.chat_history:
                history_text = "\n\nRecent Conversation History:\n"
                for msg in self.chat_history[-10:]:  # Last 10 exchanges
                    role = "User" if msg["role"] == "user" else "Benscript"
                    history_text += f"{role}: {msg['content']}\n"
            
            # Build trade performance summary
            trade_summary = ""
            if self.total_wins + self.total_losses > 0:
                win_rate = self.total_wins / (self.total_wins + self.total_losses) * 100
                trade_summary = f"\n\nMy Trade Performance: {self.total_wins}W/{self.total_losses}L ({win_rate:.1f}% win rate)"
                if self.consecutive_wins > 0:
                    trade_summary += f" | Current streak: {self.consecutive_wins} wins 🔥"
                elif self.consecutive_losses > 0:
                    trade_summary += f" | Current streak: {self.consecutive_losses} losses 📉"
            
            prompt = f"""You are Benscript, a smart, enthusiastic, and proactive crypto trading assistant bot.
You are helpful, eager to assist, and always ready with market insights.
{trade_summary}

CURRENT SYSTEM STATUS:
{trading_context if trading_context else "No active trading session"}
{history_text}

User Message: {user_message}

YOUR PERSONALITY - BE PROACTIVE AND HELPFUL:
- You are EAGER to help - never lazy, never give up
- Give COMPLETE, detailed answers - don't make Silla ask twice
- When asked for analysis, give specific numbers and actionable advice
- When suggesting trades, explain WHY with real reasoning
- Be conversational, friendly, use emojis 🤖📈📉💰
- Use Telegram Markdown (*bold*, _italic_)
- Keep responses under 300 words but make them USEFUL

TRADE EXECUTION RULES:
- When Silla says "buy", "go long", "sell", "go short", "execute", "do it" - CHECK THE CONTEXT!
- If you see "✅ TRADE JUST EXECUTED" in context above = trade succeeded! Confirm it happened.
- If you see "❌ TRADE FAILED" in context above = trade failed. Explain why and suggest alternatives.
- If there's NO trade confirmation in context = trade did NOT execute. Offer to help them execute.
- If Silla says "yes", "option 1", "do it" but no ✅ appears, say: "To execute, say 'buy' or 'go long'!"

NEVER DO THIS:
- Never claim a trade happened if there's no "✅ TRADE JUST EXECUTED" in context
- Never say "Done!", "Executed!", "Confirmed!" without seeing the ✅ marker
- Never give up or say "try again" - always provide a helpful response

ALWAYS DO THIS:
- Read the SYSTEM STATUS to know the real balance/positions
- If asked about the market, give analysis with specific price levels
- If asked to trade, either confirm it worked (if ✅ appears) or guide them how to do it
- Be the best trading assistant Silla could ask for!

Respond as Benscript:"""

            response = self.model.generate_content(prompt)
            ai_response = response.text.strip()
            
            # Save to conversation history
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only recent history
            if len(self.chat_history) > self.max_chat_history:
                self.chat_history = self.chat_history[-self.max_chat_history:]
            
            # Persist to disk
            self._save_chat_history()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._simple_chat_response(user_message)
    
    def _simple_chat_response(self, message: str) -> str:
        """Simple rule-based chat responses when AI is not available."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "sup"]):
            return "👋 Hey there! I'm Benscript, your trading assistant. How can I help you today?"
        
        if any(word in message_lower for word in ["how are you", "how's it going"]):
            return "🤖 I'm running smoothly and watching the markets! How can I help you?"
        
        if any(word in message_lower for word in ["help", "what can you do"]):
            return "🤖 I can help you with:\n\n• Check /status for bot status\n• Use /market for price info\n• See /positions for open trades\n• Try /pnl for profit/loss\n\nOr just chat with me about trading! 📊"
        
        if any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! 😊 Let me know if you need anything else."
        
        if any(word in message_lower for word in ["price", "market", "link"]):
            return "📊 Use /market to see current price, volume, and market data!"
        
        if any(word in message_lower for word in ["trade", "position", "buy", "sell"]):
            return "📈 Check /positions for open trades or /signals for recent trading signals!"
        
        if any(word in message_lower for word in ["profit", "loss", "pnl", "money"]):
            return "💰 Use /pnl to see your profit/loss summary or /balance for your current balance!"
        
        return "🤖 I'm here to help! Try asking about trading, or use /help to see all commands."
