# Julaba - AI-Enhanced Crypto Trading Bot

Automated trading bot for MEXC exchange with AI signal filtering using Google Gemini and Telegram notifications.

## 🚀 Features

- **AI-Powered Signal Filtering**: Uses Google Gemini 2.5 Flash to validate trading signals
- **Telegram Integration**: Real-time notifications and interactive bot commands
- **Paper Trading Mode**: Test strategies risk-free before going live
- **Multiple Trading Modes**: Filter, Advisory, Autonomous, and Hybrid
- **Risk Management**: Automatic stop-loss, take-profit levels, and position sizing
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and custom indicators

## 📋 Prerequisites

- **macOS**: macOS 10.15 (Catalina) or later
- **Python**: Python 3.11 or later (recommended)
- **MEXC Account**: [Sign up here](https://www.mexc.com/register)
- **Gemini API Key**: [Get free API key](https://aistudio.google.com/app/apikey)
- **Telegram Account**: For bot notifications (optional)

## 🛠️ Installation

### 1. Install Python (if not already installed)

```bash
# Check if Python is installed
python3 --version

# If not installed, download from python.org or use Homebrew:
brew install python@3.11
```

### 2. Clone or Extract the Project

```bash
cd ~/Downloads/julaba
# Or extract the ZIP file you received
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit with your favorite editor (nano, vim, or VS Code)
nano .env
```

**Required Configuration:**

```env
# MEXC API (required for live trading, optional for paper trading)
API_KEY=your_mexc_api_key
API_SECRET=your_mexc_api_secret

# Gemini API (required for AI features)
GEMINI_API_KEY=your_gemini_api_key

# Telegram Bot (optional but recommended)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

#### Getting Your API Keys:

**MEXC API Keys:**
1. Go to https://www.mexc.com/user/openapi
2. Create new API key (enable "Spot Trading" only)
3. Save the API Key and Secret

**Gemini API Key:**
1. Visit https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (it's free!)

**Telegram Bot (Optional):**
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the bot token
4. Start your bot and send a message
5. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
6. Find your `chat_id` in the response

## 🎮 Usage

### Paper Trading Mode (Recommended for Testing)

```bash
# Activate virtual environment
source venv/bin/activate

# Run in paper trading mode with $10,000 virtual balance
python bot.py --paper-balance 10000 --log-level INFO
```

### Live Trading Mode

```bash
# Make sure you have configured MEXC API keys in .env
source venv/bin/activate

# Run with INFO logging
python bot.py --log-level INFO

# Or with full DEBUG logging
python bot.py --log-level DEBUG
```

### Command Line Options

```bash
python bot.py --help

Options:
  --paper-balance AMOUNT    Run in paper trading mode with virtual balance
  --ai-confidence LEVEL     AI confidence threshold (0.0-1.0, default: 0.8)
  --ai-mode MODE           AI mode: filter, advisory, autonomous, hybrid
  --log-level LEVEL        Logging level: DEBUG, INFO, WARNING, ERROR
```

## 📱 Telegram Commands

Once the bot is running, you can control it via Telegram:

- `/status` - Bot status and performance
- `/balance` - Current balance
- `/positions` - Open positions
- `/pnl` - Profit & Loss summary
- `/ai` - AI filter statistics
- `/market` - Market information
- `/signals` - Recent trading signals
- `/aimode` - Change AI mode
- `/pause` - Pause trading
- `/resume` - Resume trading
- `/stop` - Stop the bot
- `/help` - Show all commands

You can also chat naturally with the bot to ask questions about the market or execute trades!

## 🔧 Configuration

### AI Confidence Threshold

Higher values = more conservative (fewer trades, higher quality)
- `0.7` - Moderate (default)
- `0.8` - Conservative
- `0.9` - Very conservative

### AI Trading Modes

1. **Filter Mode**: AI validates signals before execution
2. **Advisory Mode**: AI suggests trades via Telegram, you confirm
3. **Autonomous Mode**: AI trades automatically
4. **Hybrid Mode**: Combines filter + advisory

## 📊 Strategy

The bot uses a combination of:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume analysis
- AI validation via Gemini

## ⚠️ Important Notes

### Risk Management
- **Start with paper trading** to understand the bot
- **Never invest more than you can afford to lose**
- Use appropriate position sizing
- Monitor the bot regularly

### macOS-Specific
- Keep your Mac awake (System Settings > Energy Saver)
- Or use `caffeinate` command:
  ```bash
  caffeinate -i python bot.py --paper-balance 10000
  ```

### Security
- **Never share your `.env` file**
- Keep API keys secure
- Use API key restrictions (IP whitelist, trading only)
- Disable withdrawals on your MEXC API key

## 🐛 Troubleshooting

### Python Version Issues
```bash
# Make sure you're using Python 3.11+
python3 --version

# Create venv with specific version
python3.11 -m venv venv
```

### SSL Certificate Errors
```bash
# Install certificates (macOS)
/Applications/Python\ 3.11/Install\ Certificates.command
```

### Module Not Found
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Network Errors
- Check your internet connection
- Try using a VPN if Telegram is blocked
- Verify MEXC API is accessible in your region

## 📁 Project Structure

```
julaba/
├── bot.py                 # Main bot entry point
├── ai_filter.py          # AI signal filtering
├── telegram_bot.py       # Telegram integration
├── indicator.py          # Technical indicators
├── julaba_reader.py      # Data reader
├── requirements.txt      # Python dependencies
├── .env                  # Your configuration (DO NOT SHARE)
├── .env.example          # Example configuration
└── README.md             # This file
```

## 📝 Logs

Logs are saved to `julaba.log` in the project directory. Use `tail -f julaba.log` to monitor in real-time.

## 🤝 Support

If you encounter issues:
1. Check the logs (`julaba.log`)
2. Verify all API keys are correct
3. Ensure you're using Python 3.11+
4. Try paper trading mode first

## ⚖️ License

For personal use only. Not financial advice. Trade at your own risk.

---

**Happy Trading! 🚀**
