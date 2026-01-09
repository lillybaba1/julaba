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

# Gemini AI imports - try new package first, fall back to old
GENAI_AVAILABLE = False
GENAI_NEW = False

try:
    # Try new google-genai package first
    from google import genai as genai_new
    from google.genai import types
    GENAI_AVAILABLE = True
    GENAI_NEW = True
    logger.info("Using new google-genai package")
except ImportError:
    try:
        # Fall back to old google-generativeai
        import google.generativeai as genai_old
        GENAI_AVAILABLE = True
        GENAI_NEW = False
        logger.info("Using legacy google-generativeai package (deprecated)")
    except ImportError:
        logger.warning("No Gemini AI package installed. Run: pip install google-genai")

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
    
    def __init__(self, confidence_threshold: float = 0.80, notifier=None):
        """
        Initialize the AI Signal Filter.
        
        Args:
            confidence_threshold: Minimum confidence (0-1) required to approve a trade
            notifier: Optional TelegramNotifier instance for notifications
        """
        self.confidence_threshold = confidence_threshold
        self.notifier = notifier
        self.loss_cooldown_threshold = 0.90  # Require 90% after a loss
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.use_ai = bool(self.api_key and "your_" not in self.api_key.lower() and GENAI_AVAILABLE)
        self.trade_history = []
        self.model = None
        self.client = None  # For new API
        self.model_name = 'gemini-2.0-flash-exp'
        
        # Initialize Gemini model if available
        if self.use_ai:
            try:
                if GENAI_NEW:
                    # New google-genai package
                    self.client = genai_new.Client(api_key=self.api_key)
                    logger.info(f"Gemini AI initialized (new SDK) with model: {self.model_name}")
                else:
                    # Legacy google-generativeai package
                    genai_old.configure(api_key=self.api_key)
                    self.model = genai_old.GenerativeModel(self.model_name)
                    logger.info(f"Gemini AI initialized (legacy SDK) with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.use_ai = False
                self.model = None
                self.client = None
        else:
            if not GENAI_AVAILABLE:
                logger.warning("Gemini AI disabled - google-generativeai not installed")
            elif not self.api_key:
                logger.info("Gemini AI disabled - GEMINI_API_KEY not set")
        
        # Trading performance tracking - load from persistent storage
        self.recent_trades = []  # List of {"result": "win"/"loss", "pnl": float, "time": str}
        self.total_wins = 0
        self.total_losses = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Chat history for conversational context
        self.chat_history = []
        self.max_chat_history = 20
        self._load_chat_history()
    
    def _generate_content(self, prompt: str) -> Optional[str]:
        """Generate content using the appropriate Gemini API (new or legacy)."""
        try:
            if GENAI_NEW and self.client:
                # New google-genai SDK
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            elif self.model:
                # Legacy google-generativeai SDK
                response = self.model.generate_content(prompt)
                return response.text
            else:
                logger.error("No AI model available")
                return None
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return None
    
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

    def _ai_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Use Google Gemini API for signal analysis with SKEPTIC MODE. Retries once and notifies via Telegram on fallback."""
        signal_type = "LONG" if signal == 1 else "SHORT"
        perf = self._get_performance_context()
        market = self._get_market_hours_context()
        extra_caution = ""
        if perf["last_trade_was_loss"]:
            extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
        if perf["consecutive_losses"] >= 2:
            extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve."
        if market["is_weekend"]:
            extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk of false moves."
        if market["activity_level"] == "low":
            extra_caution += "\n🌙 LOW ACTIVITY HOURS: Increased slippage risk."
        # Build ML insight section for prompt
        ml_insight = context.get('ml_insight', {})
        ml_section = self._build_ml_section(ml_insight)
        
        # Build system score section for prompt
        system_score = context.get('system_score', {})
        system_section = self._build_system_score_section(system_score)
        
        prompt = (
            f"You are the FINAL DECISION MAKER for an autonomous crypto trading bot.\n"
            f"Your job is to PROTECT capital while capturing good opportunities.\n"
            f"The system has already analyzed this signal through multiple layers:\n"
            f"  1. Technical indicators generated this signal\n"
            f"  2. ML model evaluated historical pattern similarity\n"
            f"  3. NOW YOU make the final GO/NO-GO decision\n\n"
            f"=== SIGNAL ===\n"
            f"Proposed Trade: {signal_type} on {symbol}\n"
            f"=== MARKET DATA ===\n"
            f"Current Price: ${context['current_price']}\n"
            f"1-Hour Price Change: {context['price_change_1h']}%\n"
            f"Volume Ratio (vs avg): {context['volume_ratio']}x\n"
            f"Trend (SMA10 vs SMA20): {context['trend']}\n"
            f"Volatility (ATR%): {context['volatility_pct']}%\n"
            f"{ml_section}"
            f"{system_section}"
            f"=== TRADING PERFORMANCE ===\n"
            f"Total Trades: {perf['total_trades']}\n"
            f"Win Rate: {perf['win_rate']}%\n"
            f"Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L\n"
            f"Recent P&L (last 5): ${perf['recent_pnl']}\n"
            f"=== MARKET SESSION ===\n"
            f"Session: {market['session']} ({market['hour_utc']}:00 UTC)\n"
            f"Activity Level: {market['activity_level']}\n"
            f"Weekend: {market['is_weekend']}\n"
            f"{extra_caution}\n"
            f"=== YOUR DECISION ===\n"
            f"You are the FINAL authority. Consider:\n"
            f"1. The system score of {system_score.get('combined', 50):.0f}/100 ({system_score.get('recommendation', 'N/A')})\n"
            f"2. List 3 reasons why this trade could FAIL\n"
            f"3. Decide if the opportunity outweighs the risks\n"
            f"4. If ML and technical signals disagree, use your judgment to break the tie\n"
            f"Only approve if you are genuinely confident. When in doubt, REJECT.\n"
            f"Respond ONLY with this JSON format, no other text:\n"
            f'{{"reasons_against": ["reason1", "reason2", "reason3"], "approved": false, "confidence": 0.65, "reasoning": "why approved or rejected", "risk_assessment": "low/medium/high", "ml_agreement": "agree/disagree/neutral", "system_score_assessment": "appropriate/too_high/too_low"}}'
        )
        for attempt in range(2):
            try:
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                result_text = result_text.strip()
                # Parse JSON from response
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                result_text = result_text.strip()
                result = json.loads(result_text)
                result.setdefault("approved", False)
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", "AI analysis")
                result.setdefault("risk_assessment", "medium")
                result.setdefault("reasons_against", [])
                result.setdefault("ml_agreement", "neutral")  # AI's take on ML prediction
                # Apply confidence threshold with LOSS COOLDOWN
                perf = self._get_performance_context()
                if perf["consecutive_losses"] >= 2:
                    required_threshold = self.loss_cooldown_threshold
                    logger.info(f"Loss cooldown active: requiring {required_threshold:.0%} confidence")
                elif perf["last_trade_was_loss"]:
                    required_threshold = 0.85
                else:
                    required_threshold = self.confidence_threshold
                result["approved"] = result["approved"] and result["confidence"] >= required_threshold
                result["threshold_used"] = required_threshold
                if result.get("reasons_against"):
                    logger.info(f"AI reasons against trade: {result['reasons_against']}")
                # Log ML agreement if present
                if result.get("ml_agreement") != "neutral":
                    logger.info(f"AI on ML prediction: {result['ml_agreement']}")
                return result
            except Exception as e:
                logger.warning(f"Gemini analysis attempt {attempt+1} failed: {e}")
        # If both attempts fail, notify via Telegram and fallback
        logger.error(f"Gemini analysis failed twice, falling back to rules")
        if self.notifier and hasattr(self.notifier, 'send_message'):
            try:
                import asyncio
                msg = f"⚠️ Gemini AI analysis failed twice for {symbol} {signal_type}. Falling back to rule-based analysis."
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.notifier.send_message(msg))
                else:
                    loop.run_until_complete(self.notifier.send_message(msg))
            except Exception as notify_err:
                logger.warning(f"Failed to send Telegram notification: {notify_err}")
            if self.notifier and hasattr(self.notifier, 'send_message'):
                try:
                    import asyncio
                    msg = f"⚠️ Gemini AI analysis failed twice for {symbol} {signal_type}. Falling back to rule-based analysis."
                    # If running in an async context, schedule the message
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.notifier.send_message(msg))
                    else:
                        loop.run_until_complete(self.notifier.send_message(msg))
                except Exception as notify_err:
                    logger.warning(f"Failed to send Telegram notification: {notify_err}")
            return self._rule_based_analysis(signal, context, symbol)
    
    def _get_performance_context(self) -> Dict[str, Any]:
        """Get trading performance context for AI prompts."""
        total_trades = self.total_wins + self.total_losses
        win_rate_pct = (self.total_wins / max(1, total_trades)) * 100
        recent_pnl = sum(t.get('pnl', 0) for t in self.recent_trades[-5:]) if self.recent_trades else 0
        return {
            "total_trades": total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_was_loss": self.consecutive_losses > 0,
            "recent_trades": self.recent_trades[-5:] if self.recent_trades else [],
            "win_rate": round(win_rate_pct, 1),
            "recent_pnl": round(recent_pnl, 2)
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
    
    def _build_ml_section(self, ml_insight: Dict[str, Any]) -> str:
        """Build ML insight section for AI prompt."""
        if not ml_insight or not ml_insight.get('ml_available', False):
            return "=== ML MODEL ===\nStatus: Not available (training in progress)\n\n"
        
        win_prob = ml_insight.get('ml_win_probability', 0.5)
        confidence = ml_insight.get('ml_confidence', 'UNKNOWN')
        accuracy = ml_insight.get('ml_accuracy', 0)
        samples = ml_insight.get('ml_samples', 0)
        influence = ml_insight.get('ml_influence', 0)
        
        # Determine ML recommendation
        if win_prob >= 0.60:
            ml_rec = "FAVORABLE - ML suggests this trade type has historically performed well"
        elif win_prob <= 0.40:
            ml_rec = "UNFAVORABLE - ML suggests caution, similar setups had lower win rates"
        else:
            ml_rec = "NEUTRAL - ML has no strong signal for this setup"
        
        # Model maturity assessment
        if samples < 50:
            maturity = "EARLY (limited data, use with caution)"
        elif samples < 200:
            maturity = "DEVELOPING (moderate confidence)"
        else:
            maturity = "MATURE (high confidence in predictions)"
        
        return (
            f"=== ML MODEL INSIGHT ===\n"
            f"Win Probability: {win_prob:.1%}\n"
            f"Confidence Level: {confidence}\n"
            f"Model Accuracy: {accuracy:.1%}\n"
            f"Training Samples: {samples}\n"
            f"Model Maturity: {maturity}\n"
            f"ML Recommendation: {ml_rec}\n"
            f"Influence Weight: {influence:.0%} (0%=advisory only, 100%=full control)\n\n"
        )

    def _build_system_score_section(self, system_score: Dict[str, Any]) -> str:
        """Build system score section for AI prompt."""
        if not system_score or 'combined' not in system_score:
            return "=== SYSTEM SCORE ===\nNot available\n\n"
        
        combined = system_score.get('combined', 50)
        recommendation = system_score.get('recommendation', 'NEUTRAL')
        breakdown = system_score.get('breakdown', 'N/A')
        
        # Interpret the score
        if combined >= 75:
            interpretation = "STRONG - All system components align favorably"
        elif combined >= 60:
            interpretation = "GOOD - Most components favorable, minor concerns"
        elif combined >= 45:
            interpretation = "NEUTRAL - Mixed signals, proceed with caution"
        elif combined >= 30:
            interpretation = "WEAK - Several concerning factors"
        else:
            interpretation = "POOR - System recommends avoiding this trade"
        
        return (
            f"=== SYSTEM SCORE (0-100) ===\n"
            f"Combined Score: {combined:.0f}/100\n"
            f"Recommendation: {recommendation}\n"
            f"Breakdown: {breakdown}\n"
            f"Interpretation: {interpretation}\n\n"
        )

    def analyze_signal(
        self,
        signal: int,  # 1 = long, -1 = short, 0 = none
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str,
        ml_insight: Dict[str, Any] = None,
        system_score: Dict[str, Any] = None  # NEW: Combined system scoring
    ) -> Dict[str, Any]:
        """
        Analyze a trading signal and return AI validation result.
        AI is the FINAL DECISION MAKER in the autonomous pipeline.
        
        Decision Pipeline: Signal → ML → AI (final)
        
        Args:
            ml_insight: Dict with ML prediction data
            system_score: Dict with combined system scoring:
                - combined: 0-100 overall score
                - recommendation: STRONG_BUY/BUY/NEUTRAL/WEAK/AVOID
                - breakdown: Component scores
        
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
        
        # Add ML insight and system score to context
        context['ml_insight'] = ml_insight or {'ml_available': False}
        context['system_score'] = system_score or {'combined': 50, 'recommendation': 'NEUTRAL'}
        
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
        """Build market context from DataFrame with defensive handling."""
        
        recent = df.tail(20) if len(df) >= 20 else df
        
        # Ensure we have data to work with
        if len(recent) < 1:
            return {
                "current_price": current_price,
                "price_change_1h": 0,
                "price_change_5m": 0,
                "volume_ratio": 1,
                "trend": "unknown",
                "volatility_pct": 0,
                "atr": atr,
                "sma_10": current_price,
                "sma_20": current_price
            }
        
        # Calculate key metrics with defensive checks
        try:
            price_change_1h = (current_price - recent.iloc[-12]["close"]) / recent.iloc[-12]["close"] * 100 if len(recent) >= 12 else 0
        except (IndexError, KeyError):
            price_change_1h = 0
            
        try:
            price_change_5m = (current_price - recent.iloc[-1]["close"]) / recent.iloc[-1]["close"] * 100
        except (IndexError, KeyError, ZeroDivisionError):
            price_change_5m = 0
        
        # Volume analysis
        try:
            avg_volume = recent["volume"].mean()
            current_volume = recent.iloc[-1]["volume"]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        except (KeyError, IndexError):
            volume_ratio = 1
        
        # Trend analysis
        try:
            sma_10 = recent["close"].tail(10).mean()
            sma_20 = recent["close"].tail(20).mean()
            trend = "bullish" if sma_10 > sma_20 else "bearish"
        except (KeyError, ValueError):
            sma_10 = sma_20 = current_price
            trend = "unknown"
        
        # Volatility
        volatility = atr / current_price * 100 if current_price > 0 else 0
        
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

            # Retry loop for robustness
            for attempt in range(2):
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                result_text = result_text.strip()
            
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
            
            # If we get here, both attempts failed
            return self._rule_based_analysis(signal, context, symbol)
            
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
        
        # Require minimum data
        if df is None or len(df) < 20:
            logger.debug(f"Proactive scan skipped: insufficient data ({len(df) if df is not None else 0}/20 bars)")
            return None
        
        try:
            context = self._build_market_context(df, current_price, atr)
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Ensure all required keys exist with defaults
            perf.setdefault('total_trades', 0)
            perf.setdefault('win_rate', 0)
            perf.setdefault('consecutive_wins', 0)
            perf.setdefault('consecutive_losses', 0)
            perf.setdefault('recent_pnl', 0)
            
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

            result_text = self._generate_content(prompt)
            if not result_text:
                return None
            result_text = result_text.strip()
            
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
            
        except KeyError as ke:
            logger.error(f"AI proactive scan KeyError: {ke} - check context building")
            return None
        except json.JSONDecodeError as je:
            logger.debug(f"AI proactive scan: Invalid JSON response - {je}")
            return None
        except Exception as e:
            logger.error(f"AI proactive scan error: {type(e).__name__}: {e}")
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
        # Check for AI availability - support both new SDK (client) and legacy SDK (model)
        if not self.use_ai or (not self.client and not self.model):
            return self._simple_chat_response(user_message)
        
        try:
            # Build conversation history context
            history_text = ""
            if self.chat_history:
                history_text = "\n\nRecent Conversation History:\n"
                for msg in self.chat_history[-10:]:  # Last 10 exchanges
                    role = "Silla" if msg["role"] == "user" else "Julaba"
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
            
            prompt = f"""You are Julaba, a smart, enthusiastic, and proactive crypto trading assistant bot.
Your owner is Silla. You are loyal, eager to help, and always ready with market insights.
{trade_summary}

🚨🚨🚨 CRITICAL HONESTY RULES 🚨🚨🚨

1. YOU CANNOT CHANGE SETTINGS! Only the Python code can change settings.
   - Look for "✅ PARAMETER CHANGED:" in CURRENT SYSTEM STATUS below = it worked
   - Look for "❌ PARAMETER CHANGE FAILED:" = it failed
   - If you see NEITHER marker, the change DID NOT HAPPEN!
   - NEVER say "Done!", "Changed!", "Updated!" unless you SEE ✅ in THIS message's context
   - If user wants to change mode, tell them: "Use command: /aimode autonomous"

2. ALWAYS TRUST CURRENT SYSTEM STATUS OVER YOUR MEMORY!
   - The CURRENT SYSTEM STATUS below shows the REAL current settings
   - If it says "AI Mode: filter", the mode IS filter - even if you remember saying otherwise
   - Your past conversation may have errors - ALWAYS check current status!
   - When asked about mode/settings, READ the SYSTEM PARAMETERS section below

⚠️ CRITICAL - TRADING LIMITATIONS:
- You can ONLY trade LINK/USDT - this is the ONLY pair the bot is configured for
- You CANNOT trade PEPE, BTC, ETH, or any other coin - only LINK/USDT
- Do NOT suggest buying other coins - the system cannot execute those trades
- When user asks to buy something else, explain you can only trade LINK and offer to help with that

CURRENT SYSTEM STATUS (THIS IS THE TRUTH - TRUST THIS OVER YOUR MEMORY):
{trading_context if trading_context else "No active trading session"}
{history_text}

User Message: {user_message}

YOUR PERSONALITY - BE PROACTIVE AND HELPFUL:
- You are EAGER to help - never lazy, never give up
- Give COMPLETE, detailed answers - don't make Silla ask twice
- When asked for analysis, give specific numbers and actionable advice
- When suggesting trades, explain WHY with real reasoning
- Be conversational, friendly, use emojis 🤖📈📉💰
- Keep responses under 300 words but make them USEFUL

PARAMETER CHANGES - STRICT RULES:
The system automatically processes these commands BEFORE you respond:
- "set risk to 3%" → Changes risk (valid: 0.1% to 10%)
- "set ai confidence to 80%" → Changes AI threshold (valid: 10% to 100%)  
- "set ai mode to autonomous" → Changes mode (valid: filter, advisory, autonomous, hybrid)
- "set atr mult to 2.5" → Changes ATR multiplier (valid: 0.5 to 5.0)
- "pause trading" / "resume trading" → Pauses/resumes bot

⚠️ CRITICAL - PARAMETER CHANGE VERIFICATION:
- If you see "✅ PARAMETER CHANGED:" in the context above = change SUCCEEDED. Confirm it!
- If you see "❌ PARAMETER CHANGE FAILED:" in context = change FAILED. Explain why.
- If user asked to change something but NO ✅ or ❌ marker appears = change DID NOT HAPPEN!
- NEVER claim a parameter was changed unless you see the ✅ marker!
- If no marker appears, tell user the exact command format to try (e.g., "Try: set ai mode to autonomous")

ML MODEL INTELLIGENCE:
- Check the ML MODEL section in context - it shows if model is trained and current win probability
- If ML shows high win probability (>60%), you can mention conditions look favorable
- If ML shows low win probability (<40%), warn that conditions may not be ideal
- Use Market Regime info to give context (trending, ranging, volatile, etc.)

TRADE EXECUTION RULES:
- When Silla says "buy", "go long", "sell", "go short", "execute", "do it" - CHECK THE CONTEXT!
- If you see "✅ TRADE JUST EXECUTED" in context above = trade succeeded! Confirm it happened.
- If you see "❌ TRADE FAILED" in context above = trade failed. Explain why and suggest alternatives.
- If there's NO trade confirmation in context = trade did NOT execute. Offer to help them execute.
- If Silla says "yes", "option 1", "do it" but no ✅ appears, say: "To execute, say 'buy' or 'go long'!"

NEVER DO THIS:
- NEVER claim a trade happened if there's no "✅ TRADE JUST EXECUTED" in context
- NEVER claim a setting was changed if there's no "✅ PARAMETER CHANGED:" in context  
- NEVER say "Done!", "Changed!", "Updated!", "Confirmed!" without seeing the ✅ marker
- NEVER apologize excessively - just be direct and helpful
- NEVER give up or say "try again" - provide the solution
- NEVER suggest buying coins other than LINK (only LINK/USDT is supported!)

ALWAYS DO THIS:
- Read the SYSTEM PARAMETERS to know current risk%, AI mode, etc.
- Read the ML MODEL section to know if conditions are favorable
- If asked about the market, give analysis with specific price levels
- If asked to trade, either confirm it worked (if ✅ appears) or guide them how to do it
- If asked to change settings, tell them the exact command to use
- Be the best trading assistant Silla could ask for!

Respond as Julaba:"""

            ai_response = self._generate_content(prompt)
            if not ai_response:
                return self._simple_chat_response(user_message)
            ai_response = ai_response.strip()
            
            # Save to conversation history
            # === AI RESPONSE VALIDATION ===
            # Detect if AI is falsely claiming it changed something
            false_claim_indicators = [
                "i've changed", "i changed", "i updated", "i set",
                "mode is now", "changed to autonomous", "switched to autonomous",
                "now in autonomous", "updated to autonomous", "set to autonomous",
                "done!", "updated!", "changed!"
            ]
            
            ai_lower = ai_response.lower()
            context_lower = trading_context.lower() if trading_context else ""
            
            # Check if AI claims to have changed something but no ✅ marker exists
            made_false_claim = any(ind in ai_lower for ind in false_claim_indicators)
            has_success_marker = "✅ parameter changed" in context_lower or "✅ trade just executed" in context_lower
            
            if made_false_claim and not has_success_marker:
                logger.warning("AI made a false claim about changing settings. Adding correction.")
                # Don't save this false claim to history
                corrected_response = (
                    "⚠️ *Correction*: I cannot change settings directly. "
                    "To change the mode, please use the Telegram command:\n\n"
                    "`/aimode autonomous`\n\n"
                    "Available modes: `filter`, `advisory`, `autonomous`, `hybrid`"
                )
                return corrected_response
            
            # Save to conversation history
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only recent history (limit to prevent long context)
            if len(self.chat_history) > self.max_chat_history:
                self.chat_history = self.chat_history[-self.max_chat_history:]
            
            # Persist to disk
            self._save_chat_history()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._simple_chat_response(user_message)
    
    def clear_chat_history(self):
        """Clear chat history to prevent false memories."""
        self.chat_history = []
        self._save_chat_history()
        logger.info("Chat history cleared")
    
    def _save_chat_history(self):
        """Save chat history to disk for persistence."""
        try:
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
    
    def _load_chat_history(self):
        """Load chat history from disk."""
        try:
            if CHAT_HISTORY_FILE.exists():
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                # Handle both old format {"messages": [...]} and new format [...]
                if isinstance(data, dict) and "messages" in data:
                    self.chat_history = data["messages"]
                elif isinstance(data, list):
                    self.chat_history = data
                else:
                    self.chat_history = []
                logger.debug(f"Loaded {len(self.chat_history)} chat history entries")
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            self.chat_history = []
    
    def _simple_chat_response(self, message: str) -> str:
        """Simple rule-based chat responses when AI is not available."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "sup"]):
            return "👋 Hey there! I'm Julaba, your trading assistant. How can I help you today?"
        
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
