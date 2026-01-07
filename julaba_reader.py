"""
Julaba Reader: Summarizes and provides status for Julaba bot.
"""
import os
import json
from pathlib import Path

def read_julaba_status():
    """
    Reads and summarizes Julaba's persistent state if available.
    """
    base_dir = Path(__file__).parent
    # Example: look for a persistent stats or history file
    possible_files = [
        base_dir / "Julaba" / "ai_filter_history.json",
        base_dir / "Julaba" / "bot_state.json",
        base_dir / "Julaba" / "trades.json",
    ]
    found = False
    for file in possible_files:
        if file.exists():
            print(f"Found: {file.name}")
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(json.dumps(data, indent=2))
                found = True
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
    if not found:
        print("No Julaba persistent state files found.")

if __name__ == "__main__":
    read_julaba_status()
