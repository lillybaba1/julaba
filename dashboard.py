"""
Performance Dashboard for Julaba
Simple Flask-based web dashboard for monitoring trading performance.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import threading

logger = logging.getLogger("Julaba.Dashboard")

# Dashboard dependencies (optional)
try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not installed. Run: pip install flask")


# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Julaba Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
            font-size: 2.5rem;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        .section-title {
            color: #00d4ff;
            font-size: 1.2rem;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        @media (max-width: 900px) {
            .grid-3, .grid-2 { grid-template-columns: 1fr; }
        }
        .card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
        }
        .card-lg {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .card h3 {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .card .value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #fff;
        }
        .card .value.sm { font-size: 1.1rem; }
        .card .value.positive { color: #00ff88; }
        .card .value.negative { color: #ff4444; }
        .card .value.warning { color: #ffaa00; }
        .card .value.info { color: #00d4ff; }
        .card .sub { font-size: 0.8rem; color: #666; margin-top: 4px; }
        
        .indicator-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .indicator-row:last-child { border-bottom: none; }
        .indicator-name { color: #888; font-size: 0.85rem; }
        .indicator-value { font-weight: bold; font-size: 0.95rem; }
        .indicator-bar {
            width: 100px;
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-left: 10px;
        }
        .indicator-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s;
        }
        
        .signal-box {
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 10px;
        }
        .signal-box.long { background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.05)); border: 1px solid rgba(0,255,136,0.3); }
        .signal-box.short { background: linear-gradient(135deg, rgba(255,68,68,0.2), rgba(255,68,68,0.05)); border: 1px solid rgba(255,68,68,0.3); }
        .signal-box.neutral { background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.1); }
        .signal-label { font-size: 0.8rem; color: #888; margin-bottom: 5px; }
        .signal-value { font-size: 1.8rem; font-weight: bold; }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-success { background: #00ff88; color: #000; }
        .badge-danger { background: #ff4444; color: #fff; }
        .badge-warning { background: #ffaa00; color: #000; }
        .badge-info { background: #00d4ff; color: #000; }
        .badge-neutral { background: #444; color: #fff; }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .chart-container h2 {
            margin-bottom: 15px;
            color: #00d4ff;
            font-size: 1rem;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .stats-table th, .stats-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stats-table th { color: #666; font-weight: normal; font-size: 0.75rem; text-transform: uppercase; }
        .trade-row { transition: background 0.2s; }
        .trade-row:hover { background: rgba(255, 255, 255, 0.03); }
        
        .mtf-grid { display: flex; gap: 10px; margin-top: 10px; }
        .mtf-item {
            flex: 1;
            padding: 10px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            text-align: center;
        }
        .mtf-tf { font-size: 0.7rem; color: #666; margin-bottom: 5px; }
        .mtf-trend { font-size: 1rem; font-weight: bold; }
        .mtf-trend.bullish { color: #00ff88; }
        .mtf-trend.bearish { color: #ff4444; }
        .mtf-trend.neutral { color: #888; }
        
        .progress-ring {
            width: 60px;
            height: 60px;
            margin: 0 auto;
        }
        
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
        .status-dot.online { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-dot.offline { background: #ff4444; }
        .status-dot.paused { background: #ffaa00; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .tf-btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #888;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
        }
        .tf-btn:hover { background: rgba(0,212,255,0.2); color: #fff; }
        .tf-btn.active { background: #00d4ff; color: #000; border-color: #00d4ff; }
        
        /* Custom floating tooltip for chart */
        .chart-tooltip {
            position: absolute;
            display: none;
            background: rgba(10, 10, 30, 0.95);
            border: 2px solid #00d4ff;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.85rem;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
            min-width: 180px;
        }
        .chart-tooltip .tt-header {
            color: #00d4ff;
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .chart-tooltip .tt-row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }
        .chart-tooltip .tt-label {
            color: #888;
        }
        .chart-tooltip .tt-value {
            color: #fff;
            font-weight: 500;
            font-family: 'Consolas', monospace;
        }
        .chart-tooltip .tt-value.up { color: #00ff88; }
        .chart-tooltip .tt-value.down { color: #ff4444; }
        .chart-tooltip .tt-change {
            margin-top: 8px;
            padding-top: 6px;
            border-top: 1px solid rgba(255,255,255,0.1);
            text-align: center;
            font-size: 1rem;
            font-weight: bold;
        }
        
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.85rem;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
            transition: transform 0.2s;
        }
        .refresh-btn:hover { transform: scale(1.05); }
        
        .last-update { 
            position: fixed;
            bottom: 25px;
            left: 20px;
            font-size: 0.75rem;
            color: #444;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Julaba Trading Dashboard</h1>
        
        <!-- Status Row -->
        <div class="grid">
            <div class="card">
                <h3>Status</h3>
                <div class="value">
                    <span class="status-dot" id="status-dot"></span>
                    <span id="status-text">Loading...</span>
                </div>
            </div>
            <div class="card">
                <h3>Balance</h3>
                <div class="value" id="balance">$0.00</div>
                <div class="sub" id="balance-change">--</div>
            </div>
            <div class="card">
                <h3>Today's P&L</h3>
                <div class="value" id="today-pnl">$0.00</div>
            </div>
            <div class="card">
                <h3>Total P&L</h3>
                <div class="value" id="total-pnl">$0.00</div>
            </div>
            <div class="card">
                <h3>Win Rate</h3>
                <div class="value" id="win-rate">0%</div>
                <div class="sub" id="win-loss">0W / 0L</div>
            </div>
            <div class="card">
                <h3>Trades</h3>
                <div class="value" id="total-trades">0</div>
            </div>
        </div>
        
        <div class="grid-3">
            <!-- Current Signal -->
            <div class="card-lg">
                <h2 class="section-title">📡 Current Signal</h2>
                <div id="signal-display">
                    <div class="signal-box neutral">
                        <div class="signal-label">Signal Direction</div>
                        <div class="signal-value">WAIT</div>
                    </div>
                </div>
                <div id="signal-details" style="margin-top: 15px;">
                    <div class="indicator-row">
                        <span class="indicator-name">Confidence</span>
                        <span class="indicator-value" id="sig-confidence">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Entry Price</span>
                        <span class="indicator-value" id="sig-entry">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Stop Loss</span>
                        <span class="indicator-value negative" id="sig-sl">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Take Profit</span>
                        <span class="indicator-value positive" id="sig-tp">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Technical Indicators -->
            <div class="card-lg">
                <h2 class="section-title">📊 Technical Indicators</h2>
                <div id="indicators-display">
                    <div class="indicator-row">
                        <span class="indicator-name">RSI (14)</span>
                        <div style="display: flex; align-items: center;">
                            <span class="indicator-value" id="ind-rsi">--</span>
                            <div class="indicator-bar"><div class="indicator-bar-fill" id="rsi-bar" style="width: 50%; background: #00d4ff;"></div></div>
                        </div>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">MACD Signal</span>
                        <span class="indicator-value" id="ind-macd">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">ADX</span>
                        <span class="indicator-value" id="ind-adx">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">ATR</span>
                        <span class="indicator-value" id="ind-atr">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">BB Position</span>
                        <span class="indicator-value" id="ind-bb">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Volume Ratio</span>
                        <span class="indicator-value" id="ind-volume">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Market Regime -->
            <div class="card-lg">
                <h2 class="section-title">🎯 Market Regime</h2>
                <div style="text-align: center; padding: 10px 0;">
                    <span class="badge badge-info" id="regime-badge" style="font-size: 1rem; padding: 8px 16px;">LOADING</span>
                </div>
                <div id="regime-details" style="margin-top: 15px;">
                    <div class="indicator-row">
                        <span class="indicator-name">ADX Strength</span>
                        <span class="indicator-value" id="regime-adx">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Hurst Exponent</span>
                        <span class="indicator-value" id="regime-hurst">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Volatility</span>
                        <span class="indicator-value" id="regime-volatility">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Tradeable</span>
                        <span class="indicator-value" id="regime-tradeable">--</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="grid-3">
            <!-- Risk Manager -->
            <div class="card-lg">
                <h2 class="section-title">🛡️ Risk Manager</h2>
                <div id="risk-display">
                    <div class="indicator-row">
                        <span class="indicator-name">Can Trade</span>
                        <span class="indicator-value" id="risk-can-trade">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Mode</span>
                        <span class="indicator-value" id="risk-mode">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Base Risk</span>
                        <span class="indicator-value" id="risk-base">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Adjusted Risk</span>
                        <span class="indicator-value" id="risk-adjusted">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Kelly Fraction</span>
                        <span class="indicator-value" id="risk-kelly">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Daily P&L</span>
                        <span class="indicator-value" id="risk-daily">--</span>
                    </div>
                </div>
            </div>
            
            <!-- MTF Analysis -->
            <div class="card-lg">
                <h2 class="section-title">📈 Multi-Timeframe</h2>
                <div class="mtf-grid" id="mtf-grid">
                    <div class="mtf-item">
                        <div class="mtf-tf">3M</div>
                        <div class="mtf-trend neutral" id="mtf-3m">--</div>
                    </div>
                    <div class="mtf-item">
                        <div class="mtf-tf">15M</div>
                        <div class="mtf-trend neutral" id="mtf-15m">--</div>
                    </div>
                    <div class="mtf-item">
                        <div class="mtf-tf">1H</div>
                        <div class="mtf-trend neutral" id="mtf-1h">--</div>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <div class="indicator-row">
                        <span class="indicator-name">Confluence</span>
                        <span class="indicator-value info" id="mtf-confluence">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Alignment</span>
                        <span class="indicator-value" id="mtf-alignment">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Recommendation</span>
                        <span class="indicator-value" id="mtf-recommendation">--</span>
                    </div>
                </div>
            </div>
            
            <!-- AI Filter -->
            <div class="card-lg">
                <h2 class="section-title">🤖 AI Filter</h2>
                <div id="ai-display">
                    <div class="indicator-row">
                        <span class="indicator-name">AI Mode</span>
                        <span class="indicator-value info" id="ai-mode">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Threshold</span>
                        <span class="indicator-value" id="ai-threshold">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Approved</span>
                        <span class="indicator-value positive" id="ai-approved">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Rejected</span>
                        <span class="indicator-value negative" id="ai-rejected">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Approval Rate</span>
                        <span class="indicator-value" id="ai-rate">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Last Decision</span>
                        <span class="indicator-value" id="ai-last">--</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="grid-3">
            <!-- Current Position -->
            <div class="card-lg">
                <h2 class="section-title">💼 Current Position</h2>
                <div id="position-display">
                    <div class="signal-box neutral">
                        <div class="signal-value">NO POSITION</div>
                    </div>
                </div>
            </div>
            
            <!-- ML Model Status -->
            <div class="card-lg">
                <h2 class="section-title">🧠 ML Model</h2>
                <div id="ml-display">
                    <div style="text-align: center; padding: 8px 0;">
                        <span class="badge" id="ml-status-badge" style="font-size: 0.9rem; padding: 6px 14px;">LOADING</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Status</span>
                        <span class="indicator-value info" id="ml-status">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Accuracy</span>
                        <span class="indicator-value" id="ml-accuracy">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Training Samples</span>
                        <span class="indicator-value" id="ml-samples">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Influence</span>
                        <span class="indicator-value warning" id="ml-influence">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Last Prediction</span>
                        <span class="indicator-value" id="ml-last-pred">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Trading Parameters -->
            <div class="card-lg">
                <h2 class="section-title">⚙️ Trading Parameters</h2>
                <div id="params-display">
                    <div class="indicator-row">
                        <span class="indicator-name">Risk Per Trade</span>
                        <span class="indicator-value" id="param-risk">2%</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">ATR Multiplier</span>
                        <span class="indicator-value" id="param-atr">1.5x</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">TP1 / TP2 / TP3</span>
                        <span class="indicator-value" id="param-tp">1.5R / 2.5R / 4R</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Position Sizing</span>
                        <span class="indicator-value" id="param-sizing">40% / 35% / 25%</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Trail Trigger</span>
                        <span class="indicator-value" id="param-trail">1.5R</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Live Price Chart -->
        <div class="chart-container">
            <h2>📊 Live Price Chart <span id="price-display" style="float: right; font-size: 1.2rem; color: #fff;">$--</span></h2>
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <button class="tf-btn active" data-tf="1m" onclick="setTimeframe('1m')">1M</button>
                <button class="tf-btn" data-tf="3m" onclick="setTimeframe('3m')">3M</button>
                <button class="tf-btn" data-tf="5m" onclick="setTimeframe('5m')">5M</button>
                <button class="tf-btn" data-tf="15m" onclick="setTimeframe('15m')">15M</button>
            </div>
            <!-- Hover info panel -->
            <div id="chart-hover-info" style="display: flex; gap: 20px; margin-bottom: 8px; padding: 8px 12px; background: rgba(0,212,255,0.1); border-radius: 6px; font-size: 0.85rem; border: 1px solid rgba(0,212,255,0.3);">
                <span>📅 <span id="hover-time" style="color: #00d4ff;">--</span></span>
                <span>O: <span id="hover-open" style="color: #fff;">--</span></span>
                <span>H: <span id="hover-high" style="color: #00ff88;">--</span></span>
                <span>L: <span id="hover-low" style="color: #ff4444;">--</span></span>
                <span>C: <span id="hover-close" style="color: #fff;">--</span></span>
                <span id="hover-change" style="margin-left: auto;">--</span>
            </div>
            <div style="position: relative;">
                <canvas id="priceChart" height="150"></canvas>
                <!-- Floating tooltip -->
                <div id="chart-float-tooltip" class="chart-tooltip">
                    <div class="tt-header">📊 Candle Details</div>
                    <div class="tt-row"><span class="tt-label">Time:</span><span class="tt-value" id="tt-time">--</span></div>
                    <div class="tt-row"><span class="tt-label">Open:</span><span class="tt-value" id="tt-open">--</span></div>
                    <div class="tt-row"><span class="tt-label">High:</span><span class="tt-value up" id="tt-high">--</span></div>
                    <div class="tt-row"><span class="tt-label">Low:</span><span class="tt-value down" id="tt-low">--</span></div>
                    <div class="tt-row"><span class="tt-label">Close:</span><span class="tt-value" id="tt-close">--</span></div>
                    <div class="tt-change" id="tt-change">--</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.75rem; color: #666;">
                <span>High: <span id="chart-high" style="color: #00ff88;">--</span></span>
                <span>Low: <span id="chart-low" style="color: #ff4444;">--</span></span>
                <span>Volume: <span id="chart-volume" style="color: #00d4ff;">--</span></span>
            </div>
        </div>
        
        <!-- Equity Chart -->
        <div class="chart-container">
            <h2>📈 Equity Curve</h2>
            <canvas id="equityChart" height="80"></canvas>
        </div>
        
        <!-- Recent Signals -->
        <div class="chart-container">
            <h2>📡 Recent Signals</h2>
            <table class="stats-table" id="signals-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Direction</th>
                        <th>Confidence</th>
                        <th>Entry</th>
                        <th>AI Decision</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="signals-body">
                    <tr><td colspan="6">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <!-- Recent Trades -->
        <div class="chart-container">
            <h2>📊 Recent Trades</h2>
            <table class="stats-table" id="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Side</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody id="trades-body">
                    <tr><td colspan="6">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <!-- System Logs -->
        <div class="chart-container">
            <h2>📜 System Logs <button onclick="fetchLogs()" style="float:right; background:#00d4ff; color:#000; border:none; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:0.75rem;">Refresh Logs</button></h2>
            <div id="logs-container" style="max-height: 300px; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 0.75rem; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
                <div id="logs-content">Loading logs...</div>
            </div>
        </div>
    </div>
    
    <div class="last-update">Last update: <span id="last-update">--</span> | Auto-refresh: <span id="auto-refresh-status" style="color:#00ff88;">ON</span></div>
    <button class="refresh-btn" onclick="refreshAll()">🔄 Refresh</button>
    
    <script>
        let equityChart = null;
        let priceChart = null;
        let currentTimeframe = '1m';
        let autoRefresh = true;
        
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        async function fetchPriceData() {
            try {
                const response = await fetch('/api/ohlc?tf=' + currentTimeframe);
                const data = await response.json();
                updatePriceChart(data);
            } catch (error) {
                console.error('Error fetching price data:', error);
            }
        }
        
        async function fetchLogs() {
            try {
                const response = await fetch('/api/logs?count=50');
                const data = await response.json();
                updateLogs(data.logs || []);
            } catch (error) {
                console.error('Error fetching logs:', error);
                document.getElementById('logs-content').innerHTML = '<span style="color:#ff4444;">Error loading logs</span>';
            }
        }
        
        function updateLogs(logs) {
            const container = document.getElementById('logs-content');
            if (!logs || logs.length === 0) {
                container.innerHTML = '<span style="color:#666;">No logs available</span>';
                return;
            }
            
            container.innerHTML = logs.map(log => {
                let color = '#888';
                if (log.level === 'ERROR') color = '#ff4444';
                else if (log.level === 'WARNING') color = '#ffaa00';
                else if (log.level === 'INFO') color = '#00d4ff';
                
                // Highlight important events
                let msg = log.message;
                if (msg.includes('✅')) color = '#00ff88';
                else if (msg.includes('❌') || msg.includes('🚫')) color = '#ff4444';
                else if (msg.includes('🧠')) color = '#9966ff';
                else if (msg.includes('📈') || msg.includes('📊')) color = '#00d4ff';
                
                return `<div style="color:${color}; padding:2px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                    <span style="color:#666; font-size:0.7rem;">${log.time}</span> 
                    <span style="color:${color};">${msg}</span>
                </div>`;
            }).join('');
            
            // Scroll to bottom
            container.scrollTop = container.scrollHeight;
        }
        
        function setTimeframe(tf) {
            currentTimeframe = tf;
            document.querySelectorAll('.tf-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.tf-btn[data-tf="' + tf + '"]').classList.add('active');
            fetchPriceData();
        }
        
        function refreshAll() {
            fetchData();
            fetchPriceData();
            fetchLogs();
        }
        
        function updatePriceChart(data) {
            if (!data || !data.candles || data.candles.length === 0) {
                console.log('No candle data received:', data);
                return;
            }
            
            const candles = data.candles;
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            // Update current price display
            const lastCandle = candles[candles.length - 1];
            const firstCandle = candles[0];
            const priceEl = document.getElementById('price-display');
            priceEl.textContent = '$' + lastCandle.c.toFixed(4);
            const priceChange = ((lastCandle.c - firstCandle.o) / firstCandle.o * 100).toFixed(2);
            priceEl.style.color = lastCandle.c >= firstCandle.o ? '#00ff88' : '#ff4444';
            
            // Update high/low/volume
            const highs = candles.map(c => c.h);
            const lows = candles.map(c => c.l);
            const volumes = candles.map(c => c.v);
            document.getElementById('chart-high').textContent = '$' + Math.max(...highs).toFixed(4);
            document.getElementById('chart-low').textContent = '$' + Math.min(...lows).toFixed(4);
            document.getElementById('chart-volume').textContent = volumes.reduce((a,b) => a+b, 0).toLocaleString();
            
            // Prepare candlestick data - convert timestamp to Date object
            const ohlcData = candles.map(c => ({
                x: new Date(c.t),
                o: c.o,
                h: c.h,
                l: c.l,
                c: c.c
            }));
            
            // Prepare line data (close prices for overlay)
            const closeData = candles.map(c => ({ x: new Date(c.t), y: c.c }));
            
            if (priceChart) {
                priceChart.data.datasets[0].data = ohlcData;
                priceChart.data.datasets[1].data = closeData;
                priceChart.update('none');
            } else {
                priceChart = new Chart(ctx, {
                    type: 'candlestick',
                    data: {
                        datasets: [{
                            label: 'Price',
                            data: ohlcData,
                            color: {
                                up: '#00ff88',
                                down: '#ff4444',
                                unchanged: '#888'
                            },
                            borderColor: {
                                up: '#00ff88',
                                down: '#ff4444',
                                unchanged: '#888'
                            }
                        }, {
                            label: 'Close',
                            type: 'line',
                            data: closeData,
                            borderColor: 'rgba(0, 212, 255, 0.5)',
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: false,
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        interaction: {
                            mode: 'index',
                            intersect: false
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: { enabled: false }  // Using custom floating tooltip instead
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: { unit: 'minute', displayFormats: { minute: 'HH:mm' } },
                                grid: { color: 'rgba(255,255,255,0.05)' },
                                ticks: { color: '#666', maxTicksLimit: 10 }
                            },
                            y: {
                                position: 'right',
                                grid: { color: 'rgba(255,255,255,0.05)' },
                                ticks: { color: '#666', callback: v => '$' + v.toFixed(2) }
                            }
                        },
                        onHover: function(event, elements) {
                            if (elements.length > 0 && elements[0].element) {
                                const dataIndex = elements[0].index;
                                const dataset = priceChart.data.datasets[0].data;
                                if (dataset && dataset[dataIndex]) {
                                    const d = dataset[dataIndex];
                                    const date = new Date(d.x);
                                    document.getElementById('hover-time').textContent = date.toLocaleTimeString();
                                    document.getElementById('hover-open').textContent = '$' + d.o.toFixed(4);
                                    document.getElementById('hover-high').textContent = '$' + d.h.toFixed(4);
                                    document.getElementById('hover-low').textContent = '$' + d.l.toFixed(4);
                                    document.getElementById('hover-close').textContent = '$' + d.c.toFixed(4);
                                    const change = ((d.c - d.o) / d.o * 100);
                                    const changeEl = document.getElementById('hover-change');
                                    changeEl.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                                    changeEl.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                                }
                            }
                        }
                    }
                });
                
                // Add mouse move listener for floating tooltip
                const canvas = document.getElementById('priceChart');
                const floatTooltip = document.getElementById('chart-float-tooltip');
                
                canvas.addEventListener('mousemove', function(e) {
                    const rect = canvas.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const elements = priceChart.getElementsAtEventForMode(e, 'index', { intersect: false }, false);
                    
                    if (elements.length > 0) {
                        const dataIndex = elements[0].index;
                        const dataset = priceChart.data.datasets[0].data;
                        if (dataset && dataset[dataIndex]) {
                            const d = dataset[dataIndex];
                            const date = new Date(d.x);
                            const change = ((d.c - d.o) / d.o * 100);
                            
                            // Update header info bar
                            document.getElementById('hover-time').textContent = date.toLocaleTimeString();
                            document.getElementById('hover-open').textContent = '$' + d.o.toFixed(4);
                            document.getElementById('hover-high').textContent = '$' + d.h.toFixed(4);
                            document.getElementById('hover-low').textContent = '$' + d.l.toFixed(4);
                            document.getElementById('hover-close').textContent = '$' + d.c.toFixed(4);
                            const changeEl = document.getElementById('hover-change');
                            changeEl.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                            changeEl.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                            
                            // Update floating tooltip
                            document.getElementById('tt-time').textContent = date.toLocaleString();
                            document.getElementById('tt-open').textContent = '$' + d.o.toFixed(4);
                            document.getElementById('tt-high').textContent = '$' + d.h.toFixed(4);
                            document.getElementById('tt-low').textContent = '$' + d.l.toFixed(4);
                            document.getElementById('tt-close').textContent = '$' + d.c.toFixed(4);
                            document.getElementById('tt-close').className = 'tt-value ' + (d.c >= d.o ? 'up' : 'down');
                            const ttChange = document.getElementById('tt-change');
                            ttChange.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                            ttChange.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                            
                            // Position tooltip near cursor
                            let tooltipX = x + 15;
                            let tooltipY = y + 15;
                            
                            // Keep tooltip within canvas bounds
                            if (tooltipX + 200 > rect.width) {
                                tooltipX = x - 200;
                            }
                            if (tooltipY + 180 > rect.height) {
                                tooltipY = y - 180;
                            }
                            
                            floatTooltip.style.left = tooltipX + 'px';
                            floatTooltip.style.top = tooltipY + 'px';
                            floatTooltip.style.display = 'block';
                        }
                    }
                });
                
                canvas.addEventListener('mouseleave', function() {
                    floatTooltip.style.display = 'none';
                });
            }
        }
        
        function updateDashboard(data) {
            // Status
            const statusDot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            if (data.status && data.status.connected) {
                if (data.status.paused) {
                    statusDot.className = 'status-dot paused';
                    statusText.textContent = 'PAUSED';
                } else {
                    statusDot.className = 'status-dot online';
                    statusText.textContent = 'RUNNING';
                }
            } else {
                statusDot.className = 'status-dot offline';
                statusText.textContent = 'OFFLINE';
            }
            
            // Balance
            const bal = data.balance || {};
            document.getElementById('balance').textContent = '$' + (bal.current || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            const changePct = bal.change_pct || 0;
            const changeEl = document.getElementById('balance-change');
            changeEl.textContent = (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%';
            changeEl.style.color = changePct >= 0 ? '#00ff88' : '#ff4444';
            
            // P&L
            const pnl = data.pnl || {};
            const todayPnl = pnl.today || 0;
            const todayEl = document.getElementById('today-pnl');
            todayEl.textContent = (todayPnl >= 0 ? '+$' : '-$') + Math.abs(todayPnl).toFixed(2);
            todayEl.className = 'value ' + (todayPnl >= 0 ? 'positive' : 'negative');
            
            const totalPnl = pnl.total || 0;
            const totalEl = document.getElementById('total-pnl');
            totalEl.textContent = (totalPnl >= 0 ? '+$' : '-$') + Math.abs(totalPnl).toFixed(2);
            totalEl.className = 'value ' + (totalPnl >= 0 ? 'positive' : 'negative');
            
            // Win Rate
            document.getElementById('win-rate').textContent = ((pnl.win_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('win-loss').textContent = (pnl.winning || 0) + 'W / ' + ((pnl.trades || 0) - (pnl.winning || 0)) + 'L';
            document.getElementById('total-trades').textContent = pnl.trades || 0;
            
            // Current Signal
            const signal = data.current_signal || {};
            const sigBox = document.querySelector('#signal-display .signal-box');
            const sigValue = document.querySelector('#signal-display .signal-value');
            if (signal.direction) {
                sigBox.className = 'signal-box ' + (signal.direction === 'LONG' ? 'long' : signal.direction === 'SHORT' ? 'short' : 'neutral');
                sigValue.textContent = signal.direction;
                document.getElementById('sig-confidence').textContent = ((signal.confidence || 0) * 100).toFixed(0) + '%';
                document.getElementById('sig-entry').textContent = signal.entry ? '$' + signal.entry.toFixed(4) : '--';
                document.getElementById('sig-sl').textContent = signal.stop_loss ? '$' + signal.stop_loss.toFixed(4) : '--';
                document.getElementById('sig-tp').textContent = signal.take_profit ? '$' + signal.take_profit.toFixed(4) : '--';
            } else {
                sigBox.className = 'signal-box neutral';
                sigValue.textContent = 'WAITING';
            }
            
            // Technical Indicators
            const ind = data.indicators || {};
            document.getElementById('ind-rsi').textContent = ind.rsi ? ind.rsi.toFixed(1) : '--';
            if (ind.rsi) {
                const rsiBar = document.getElementById('rsi-bar');
                rsiBar.style.width = ind.rsi + '%';
                rsiBar.style.background = ind.rsi > 70 ? '#ff4444' : ind.rsi < 30 ? '#00ff88' : '#00d4ff';
            }
            document.getElementById('ind-macd').textContent = ind.macd_signal || '--';
            document.getElementById('ind-adx').textContent = ind.adx ? ind.adx.toFixed(1) : '--';
            document.getElementById('ind-atr').textContent = ind.atr ? '$' + ind.atr.toFixed(4) : '--';
            document.getElementById('ind-bb').textContent = ind.bb_position || '--';
            document.getElementById('ind-volume').textContent = ind.volume_ratio ? ind.volume_ratio.toFixed(2) + 'x' : '--';
            
            // Market Regime
            const regime = data.regime || {};
            const regimeBadge = document.getElementById('regime-badge');
            regimeBadge.textContent = regime.regime || 'UNKNOWN';
            regimeBadge.className = 'badge ' + (regime.regime === 'TRENDING' || regime.regime === 'STRONG_TRENDING' ? 'badge-success' : regime.regime === 'CHOPPY' ? 'badge-danger' : 'badge-info');
            document.getElementById('regime-adx').textContent = regime.adx ? regime.adx.toFixed(1) : '--';
            document.getElementById('regime-hurst').textContent = regime.hurst ? regime.hurst.toFixed(3) : '--';
            document.getElementById('regime-volatility').textContent = regime.volatility || '--';
            document.getElementById('regime-tradeable').textContent = regime.tradeable ? '✅ YES' : '❌ NO';
            document.getElementById('regime-tradeable').className = 'indicator-value ' + (regime.tradeable ? 'positive' : 'negative');
            
            // Risk Manager
            const risk = data.risk || {};
            document.getElementById('risk-can-trade').textContent = risk.can_trade ? '✅ YES' : '🛑 NO';
            document.getElementById('risk-can-trade').className = 'indicator-value ' + (risk.can_trade ? 'positive' : 'negative');
            document.getElementById('risk-mode').textContent = risk.dd_mode || 'NORMAL';
            document.getElementById('risk-base').textContent = ((risk.base_risk || 0.02) * 100).toFixed(1) + '%';
            document.getElementById('risk-adjusted').textContent = ((risk.adjusted_risk || 0.02) * 100).toFixed(2) + '%';
            document.getElementById('risk-kelly').textContent = ((risk.kelly_risk || 0.02) * 100).toFixed(2) + '%';
            const dailyPnl = risk.daily_pnl || 0;
            document.getElementById('risk-daily').textContent = (dailyPnl >= 0 ? '+$' : '-$') + Math.abs(dailyPnl).toFixed(2);
            document.getElementById('risk-daily').className = 'indicator-value ' + (dailyPnl >= 0 ? 'positive' : 'negative');
            
            // MTF Analysis
            const mtf = data.mtf || {};
            updateMTF('mtf-3m', mtf.primary?.trend?.direction || '--');
            updateMTF('mtf-15m', mtf.secondary?.trend?.direction || '--');
            updateMTF('mtf-1h', mtf.higher?.trend?.direction || '--');
            document.getElementById('mtf-confluence').textContent = (mtf.confluence_pct || 0) + '%';
            document.getElementById('mtf-alignment').textContent = (mtf.alignment_score || 0).toFixed(2);
            document.getElementById('mtf-recommendation').textContent = mtf.recommendation || '--';
            
            // AI Filter
            const ai = data.ai || {};
            document.getElementById('ai-mode').textContent = ai.mode || 'filter';
            document.getElementById('ai-threshold').textContent = ((ai.threshold || 0.7) * 100).toFixed(0) + '%';
            document.getElementById('ai-approved').textContent = ai.approved || 0;
            document.getElementById('ai-rejected').textContent = ai.rejected || 0;
            const total = (ai.approved || 0) + (ai.rejected || 0);
            document.getElementById('ai-rate').textContent = total > 0 ? ((ai.approved || 0) / total * 100).toFixed(0) + '%' : '--';
            document.getElementById('ai-last').textContent = ai.last_decision || '--';
            
            // ML Model Status
            const ml = data.ml || {};
            const mlBadge = document.getElementById('ml-status-badge');
            if (ml.loaded) {
                mlBadge.textContent = 'ACTIVE';
                mlBadge.className = 'badge badge-success';
                document.getElementById('ml-status').textContent = ml.status || 'Loaded';
                document.getElementById('ml-accuracy').textContent = ((ml.accuracy || 0) * 100).toFixed(1) + '%';
                document.getElementById('ml-samples').textContent = ml.samples || 0;
                document.getElementById('ml-influence').textContent = ((ml.influence || 0) * 100).toFixed(0) + '% (Advisory)';
                if (ml.last_prediction) {
                    document.getElementById('ml-last-pred').textContent = 
                        ((ml.last_prediction.probability || 0.5) * 100).toFixed(0) + '% win';
                }
            } else {
                mlBadge.textContent = 'NOT LOADED';
                mlBadge.className = 'badge badge-neutral';
                document.getElementById('ml-status').textContent = 'Not Available';
                document.getElementById('ml-accuracy').textContent = '--';
                document.getElementById('ml-samples').textContent = '--';
                document.getElementById('ml-influence').textContent = '0%';
                document.getElementById('ml-last-pred').textContent = '--';
            }
            
            // Position
            const posDisplay = document.getElementById('position-display');
            const pos = data.position;
            if (pos && pos.symbol) {
                const pnlClass = (pos.pnl || 0) >= 0 ? 'positive' : 'negative';
                const boxClass = pos.side === 'LONG' ? 'long' : 'short';
                posDisplay.innerHTML = `
                    <div class="signal-box ${boxClass}">
                        <div class="signal-label">${pos.symbol}</div>
                        <div class="signal-value">${pos.side}</div>
                    </div>
                    <div style="margin-top: 10px;">
                        <div class="indicator-row">
                            <span class="indicator-name">Entry</span>
                            <span class="indicator-value">$${pos.entry.toFixed(4)}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-name">Size</span>
                            <span class="indicator-value">${pos.size.toFixed(4)}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-name">Unrealized P&L</span>
                            <span class="indicator-value ${pnlClass}">${pos.pnl >= 0 ? '+' : ''}$${pos.pnl.toFixed(2)}</span>
                        </div>
                        <div class="indicator-row">
                            <span class="indicator-name">Stop Loss</span>
                            <span class="indicator-value negative">$${(pos.stop_loss || 0).toFixed(4)}</span>
                        </div>
                    </div>
                `;
            } else {
                posDisplay.innerHTML = '<div class="signal-box neutral"><div class="signal-value">NO POSITION</div></div>';
            }
            
            // Trading Parameters
            const params = data.params || {};
            document.getElementById('param-risk').textContent = ((params.risk_pct || 0.02) * 100).toFixed(1) + '%';
            document.getElementById('param-atr').textContent = (params.atr_mult || 1.5) + 'x';
            document.getElementById('param-tp').textContent = (params.tp1_r || 1.5) + 'R / ' + (params.tp2_r || 2.5) + 'R / ' + (params.tp3_r || 4) + 'R';
            document.getElementById('param-sizing').textContent = ((params.tp1_pct || 0.4) * 100) + '% / ' + ((params.tp2_pct || 0.35) * 100) + '% / ' + ((params.tp3_pct || 0.25) * 100) + '%';
            document.getElementById('param-trail').textContent = (params.trail_trigger_r || 1.5) + 'R';
            
            // Signals Table
            const signalsBody = document.getElementById('signals-body');
            const signals = data.signals || [];
            if (signals.length === 0) {
                signalsBody.innerHTML = '<tr><td colspan="6" style="color: #666;">No signals yet</td></tr>';
            } else {
                signalsBody.innerHTML = signals.slice(-10).reverse().map(sig => {
                    const dirClass = sig.direction === 'LONG' ? 'positive' : sig.direction === 'SHORT' ? 'negative' : '';
                    const statusBadge = sig.executed ? '<span class="badge badge-success">EXECUTED</span>' : 
                                       sig.rejected ? '<span class="badge badge-danger">REJECTED</span>' : 
                                       '<span class="badge badge-neutral">PENDING</span>';
                    return `
                        <tr class="trade-row">
                            <td>${sig.time || 'N/A'}</td>
                            <td class="${dirClass}">${sig.direction || 'N/A'}</td>
                            <td>${((sig.confidence || 0) * 100).toFixed(0)}%</td>
                            <td>$${(sig.entry || 0).toFixed(4)}</td>
                            <td>${sig.ai_decision || '--'}</td>
                            <td>${statusBadge}</td>
                        </tr>
                    `;
                }).join('');
            }
            
            // Trades Table
            const tradesBody = document.getElementById('trades-body');
            const trades = data.trades || [];
            if (trades.length === 0) {
                tradesBody.innerHTML = '<tr><td colspan="6" style="color: #666;">No trades yet</td></tr>';
            } else {
                tradesBody.innerHTML = trades.slice(-10).reverse().map(trade => {
                    const pnlClass = (trade.pnl || 0) >= 0 ? 'positive' : 'negative';
                    const resultBadge = (trade.pnl || 0) >= 0 ? 
                        '<span class="badge badge-success">WIN</span>' : 
                        '<span class="badge badge-danger">LOSS</span>';
                    return `
                        <tr class="trade-row">
                            <td>${trade.time || 'N/A'}</td>
                            <td>${trade.side || 'N/A'}</td>
                            <td>$${(trade.entry || 0).toFixed(4)}</td>
                            <td>$${(trade.exit || 0).toFixed(4)}</td>
                            <td class="${pnlClass}">${trade.pnl >= 0 ? '+' : ''}$${(trade.pnl || 0).toFixed(2)}</td>
                            <td>${resultBadge}</td>
                        </tr>
                    `;
                }).join('');
            }
            
            // Equity Chart
            updateEquityChart(data.equity_curve || []);
        }
        
        function updateMTF(id, direction) {
            const el = document.getElementById(id);
            el.textContent = direction.toUpperCase();
            el.className = 'mtf-trend ' + (direction === 'bullish' ? 'bullish' : direction === 'bearish' ? 'bearish' : 'neutral');
        }
        
        function updateEquityChart(equityCurve) {
            const ctx = document.getElementById('equityChart').getContext('2d');
            if (equityCurve.length === 0) return;
            
            if (equityChart) {
                equityChart.data.labels = equityCurve.map((_, i) => i);
                equityChart.data.datasets[0].data = equityCurve;
                equityChart.update();
            } else {
                equityChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: equityCurve.map((_, i) => i),
                        datasets: [{
                            label: 'Equity',
                            data: equityCurve,
                            borderColor: '#00d4ff',
                            backgroundColor: 'rgba(0, 212, 255, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { display: false },
                            y: {
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#666', callback: v => '$' + v.toLocaleString() }
                            }
                        }
                    }
                });
            }
        }
        
        // Initial data fetch
        fetchData();
        fetchPriceData();
        fetchLogs();
        
        // Auto-refresh intervals
        setInterval(fetchData, 5000);        // Main data every 5 seconds
        setInterval(fetchPriceData, 3000);   // Price chart every 3 seconds
        setInterval(fetchLogs, 10000);       // Logs every 10 seconds
    </script>
</body>
</html>
"""


class Dashboard:
    """
    Web-based Performance Dashboard for Julaba.
    
    Provides real-time monitoring of:
    - Account balance and P&L
    - Open positions
    - Trade history
    - Market regime
    - AI filter statistics
    - Technical indicators
    - Risk management
    - Multi-timeframe analysis
    - ML model status
    - System logs
    """
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = None
        self.server_thread = None
        self.running = False
        
        # Data callbacks (set by main bot)
        self.get_status = None
        self.get_balance = None
        self.get_pnl = None
        self.get_position = None
        self.get_trades = None
        self.get_regime = None
        self.get_ai_stats = None
        self.get_equity_curve = None
        # Enhanced callbacks
        self.get_indicators = None
        self.get_current_signal = None
        self.get_risk_stats = None
        self.get_mtf_analysis = None
        self.get_params = None
        self.get_signals = None
        self.get_ohlc_data = None  # For live price chart
        # NEW: ML and logs callbacks
        self.get_ml_status = None
        self.get_system_logs = None
        
        if FLASK_AVAILABLE:
            self._setup_flask()
    
    def _setup_flask(self):
        """Setup Flask application."""
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)
        
        # Suppress Flask logs
        import logging as flask_logging
        flask_logging.getLogger('werkzeug').setLevel(flask_logging.WARNING)
        
        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/data')
        def api_data():
            data = {}
            
            try:
                if self.get_status:
                    data['status'] = self.get_status()
                if self.get_balance:
                    data['balance'] = self.get_balance()
                if self.get_pnl:
                    data['pnl'] = self.get_pnl()
                if self.get_position:
                    data['position'] = self.get_position()
                if self.get_trades:
                    data['trades'] = self.get_trades()
                if self.get_regime:
                    data['regime'] = self.get_regime()
                if self.get_ai_stats:
                    data['ai'] = self.get_ai_stats()
                if self.get_equity_curve:
                    data['equity_curve'] = self.get_equity_curve()
                # NEW: Enhanced data
                if self.get_indicators:
                    data['indicators'] = self.get_indicators()
                if self.get_current_signal:
                    data['current_signal'] = self.get_current_signal()
                if self.get_risk_stats:
                    data['risk'] = self.get_risk_stats()
                if self.get_mtf_analysis:
                    data['mtf'] = self.get_mtf_analysis()
                if self.get_params:
                    data['params'] = self.get_params()
                if self.get_signals:
                    data['signals'] = self.get_signals()
                # NEW: ML and Logs
                if self.get_ml_status:
                    data['ml'] = self.get_ml_status()
            except Exception as e:
                logger.error(f"Dashboard API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/ohlc')
        def api_ohlc():
            """Get OHLC candlestick data for live chart."""
            data = {'candles': []}
            tf = request.args.get('tf', '1m')
            
            try:
                if self.get_ohlc_data:
                    candles = self.get_ohlc_data(tf)
                    data['candles'] = candles
                    data['timeframe'] = tf
            except Exception as e:
                logger.error(f"Dashboard OHLC API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/logs')
        def api_logs():
            """Get system logs for dashboard."""
            data = {'logs': []}
            count = request.args.get('count', 50, type=int)
            
            try:
                if self.get_system_logs:
                    data['logs'] = self.get_system_logs(count)
            except Exception as e:
                logger.error(f"Dashboard logs API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
    
    def start(self):
        """Start the dashboard server in a background thread."""
        if not FLASK_AVAILABLE:
            logger.warning("Cannot start dashboard - Flask not installed")
            return False
        
        if self.running:
            return True
        
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        logger.info(f"📊 Dashboard started at http://localhost:{self.port}")
        return True
    
    def stop(self):
        """Stop the dashboard server."""
        self.running = False
        logger.info("Dashboard stopped")


# Singleton instance
_dashboard: Optional[Dashboard] = None


def get_dashboard(port: int = 5000) -> Dashboard:
    """Get the global dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = Dashboard(port=port)
    return _dashboard
