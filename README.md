# Benscript

Benscript is an advanced trading bot system with AI-powered signal filtering, Telegram integration, and indicator-based strategies. This folder contains the main components for running and managing the Benscript bot.

## Main Files

- **bot.py**: The main entry point for running the Benscript trading bot. Handles trading logic, state management, and integration with other modules.
- **ai_filter.py**: Provides the `AISignalFilter` class for AI-based trade signal filtering and performance tracking.
- **indicator.py**: Contains the `generate_signals` function for technical indicator-based signal generation.
- **telegram_bot.py**: Implements Telegram bot integration for notifications, commands, and user interaction.
- **benscript_reader.py**: Utility script to read and print Benscript's persistent state/history files for quick review.

## How to Run

1. **Install dependencies**
   - Run `pip install -r requirements.txt` in the `Benscript` directory.

2. **Set up environment**
   - Create a `.env` file with your API keys and configuration as needed by the bot.

3. **Run the bot**
   - Execute `python bot.py` to start the Benscript trading bot.

4. **Read Benscript state**
   - Run `python benscript_reader.py` to print the contents of Benscript's persistent state/history files (if any).

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python package dependencies.

## Notes
- The bot uses AI and rule-based logic for trade filtering.
- Telegram integration allows for real-time notifications and control.
- Persistent state/history files may include trade logs, AI filter history, and bot state.

---
For more details, review the code and comments in each file.