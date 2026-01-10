# Julaba

Julaba is an advanced trading bot system with AI-powered signal filtering, Telegram integration, and indicator-based strategies. This folder contains the main components for running and managing the Julaba bot.

## Main Files

- **bot.py**: The main entry point for running the Julaba trading bot. Handles trading logic, state management, and integration with other modules.
- **ai_filter.py**: Provides the `AISignalFilter` class for AI-based trade signal filtering and performance tracking.
- **indicator.py**: Contains the `generate_signals` function for technical indicator-based signal generation.
- **telegram_bot.py**: Implements Telegram bot integration for notifications, commands, and user interaction.
- **julaba_reader.py**: Utility script to read and print Julaba's persistent state/history files for quick review.

## How to Run

1. **Install dependencies**
   - Run `pip install -r requirements.txt` in the `Julaba` directory.

2. **Set up environment**
   - Create a `.env` file with your API keys and configuration as needed by the bot.

3. **Install as systemd service** (Recommended)
   - Run `sudo bash install_service.sh` to install as a system service.
   - The bot will auto-start on boot and auto-restart on crashes.

4. **Manual run** (Alternative)
   - Execute `python bot.py --dashboard` to start with web dashboard.

## Quick Commands

The bot includes a simple command tool for quick management:

```bash
j s          # Service status
j start      # Start bot
j stop       # Stop bot
j r          # Restart bot
j l          # Live logs
j la         # AI logs only
j lt         # Trade logs only

j b          # Check balance
j p          # Check position
j ai         # Check AI mode
j pnl        # Show P&L
j perf       # Performance metrics

j ml         # ML model stats
j trades     # Last trades
j err        # Recent errors
j check      # Health check
j backup     # Run backup
```

Type `j` without arguments to see all available commands.

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python package dependencies.

## Notes
- The bot uses AI and rule-based logic for trade filtering.
- Telegram integration allows for real-time notifications and control.
- Persistent state/history files may include trade logs, AI filter history, and bot state.

---
For more details, review the code and comments in each file.