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
    <meta name="viewport" content="width=1400, initial-scale=0.5, maximum-scale=2.0, user-scalable=yes">
    <meta name="HandheldFriendly" content="false">
    <meta name="MobileOptimized" content="1400">
    <title>Julaba Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        /* Animated Background */
        body {
            font-family: 'Rajdhani', 'Segoe UI', sans-serif;
            background: #000;
            color: #eee;
            min-height: 100vh;
            min-width: 1400px;
            overflow-x: auto;
            position: relative;
        }
        
        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        
        .main-content {
            position: relative;
            z-index: 1;
            padding: 20px;
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        /* 3D Title */
        h1 {
            text-align: center;
            margin-bottom: 40px;
            font-family: 'Orbitron', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00d4ff, #00ff88, #00d4ff);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 3s ease infinite;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5));
            transform: perspective(500px) rotateX(5deg);
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .section-title {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            font-size: 1.1rem;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid transparent;
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.5), transparent);
            background-size: 100% 2px;
            background-position: bottom;
            background-repeat: no-repeat;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
            perspective: 1000px;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 25px;
            perspective: 1000px;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 25px;
            perspective: 1000px;
        }
        /* Desktop mode forced on all devices - no mobile layout */
        
        /* 3D Glass Cards */
        .card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02));
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            backdrop-filter: blur(20px);
            transform-style: preserve-3d;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 0 30px rgba(0, 212, 255, 0.05);
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .card:hover {
            transform: translateY(-10px) rotateX(5deg) rotateY(-2deg) scale(1.02);
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow: 
                0 20px 50px rgba(0, 0, 0, 0.5),
                0 0 30px rgba(0, 212, 255, 0.3),
                inset 0 0 30px rgba(0, 212, 255, 0.1);
        }
        
        .card:hover::before {
            left: 100%;
        }
        
        .card-lg {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            backdrop-filter: blur(20px);
            transform-style: preserve-3d;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 0 30px rgba(0, 212, 255, 0.03);
        }
        
        .card-lg:hover {
            transform: translateY(-5px) scale(1.01);
            border-color: rgba(0, 212, 255, 0.4);
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.4),
                0 0 25px rgba(0, 212, 255, 0.2);
        }
        
        .card h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .card .value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.6rem;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            animation: valueGlow 2s ease-in-out infinite alternate;
        }
        
        @keyframes valueGlow {
            from { text-shadow: 0 0 10px rgba(255, 255, 255, 0.3); }
            to { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5), 0 0 30px rgba(0, 212, 255, 0.3); }
        }
        
        .card .value.sm { font-size: 1.1rem; }
        .card .value.positive { color: #00ff88; text-shadow: 0 0 15px rgba(0, 255, 136, 0.5); }
        .card .value.negative { color: #ff4444; text-shadow: 0 0 15px rgba(255, 68, 68, 0.5); }
        .card .value.warning { color: #ffaa00; text-shadow: 0 0 15px rgba(255, 170, 0, 0.5); }
        .card .value.info { color: #00d4ff; text-shadow: 0 0 15px rgba(0, 212, 255, 0.5); }
        .card .sub { font-size: 0.8rem; color: #666; margin-top: 4px; }
        
        .indicator-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(0, 212, 255, 0.1);
            transition: all 0.3s;
        }
        .indicator-row:hover {
            background: rgba(0, 212, 255, 0.05);
            padding-left: 10px;
            border-radius: 8px;
        }
        .indicator-row:last-child { border-bottom: none; }
        .indicator-name { color: #c9a8b3; font-size: 0.85rem; }
        .indicator-value { 
            font-family: 'Orbitron', sans-serif;
            font-weight: bold; 
            font-size: 0.95rem; 
        }
        .indicator-bar {
            width: 100px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-left: 10px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }
        .indicator-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 0 10px currentColor;
        }
        
        /* 3D Signal Box */
        .signal-box {
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 15px;
            transform-style: preserve-3d;
            transition: all 0.4s;
            position: relative;
            overflow: hidden;
        }
        .signal-box::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: signalShine 3s infinite;
        }
        @keyframes signalShine {
            0% { transform: translateX(-100%) rotate(45deg); }
            100% { transform: translateX(100%) rotate(45deg); }
        }
        .signal-box.long { 
            background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.05)); 
            border: 2px solid rgba(0,255,136,0.4);
            box-shadow: 0 0 30px rgba(0,255,136,0.2), inset 0 0 30px rgba(0,255,136,0.1);
        }
        .signal-box.short { 
            background: linear-gradient(135deg, rgba(255,68,68,0.2), rgba(255,68,68,0.05)); 
            border: 2px solid rgba(255,68,68,0.4);
            box-shadow: 0 0 30px rgba(255,68,68,0.2), inset 0 0 30px rgba(255,68,68,0.1);
        }
        .signal-box.neutral { 
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.02)); 
            border: 2px solid rgba(255,255,255,0.2);
        }
        .signal-label { font-size: 0.8rem; color: #888; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px; }
        .signal-value { 
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem; 
            font-weight: bold;
            text-shadow: 0 0 20px currentColor;
        }
        
        /* Animated Badges */
        .badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
            animation: badgePulse 2s infinite;
        }
        @keyframes badgePulse {
            0%, 100% { box-shadow: 0 0 5px currentColor; }
            50% { box-shadow: 0 0 20px currentColor; }
        }
        .badge-success { background: linear-gradient(135deg, #00ff88, #00cc6a); color: #000; }
        .badge-danger { background: linear-gradient(135deg, #ff4444, #cc2222); color: #fff; }
        .badge-warning { background: linear-gradient(135deg, #ffaa00, #cc8800); color: #000; }
        .badge-info { background: linear-gradient(135deg, #00d4ff, #0099cc); color: #000; }
        .badge-neutral { background: linear-gradient(135deg, #555, #333); color: #fff; }
        
        /* 3D Chart Container */
        .chart-container {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            backdrop-filter: blur(20px);
            transform-style: preserve-3d;
            transition: all 0.4s;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 0 30px rgba(0, 212, 255, 0.03);
        }
        .chart-container:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.4),
                0 0 30px rgba(0, 212, 255, 0.15);
        }
        .chart-container h2 {
            font-family: 'Orbitron', sans-serif;
            margin-bottom: 20px;
            color: #00d4ff;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .stats-table th, .stats-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(0, 212, 255, 0.1);
        }
        .stats-table th { 
            color: #00d4ff; 
            font-family: 'Orbitron', sans-serif;
            font-weight: normal; 
            font-size: 0.7rem; 
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .trade-row { 
            transition: all 0.3s; 
        }
        .trade-row:hover { 
            background: rgba(0, 212, 255, 0.1);
            transform: scale(1.01);
        }
        
        .mtf-grid { display: flex; gap: 15px; margin-top: 15px; }
        .mtf-item {
            flex: 1;
            padding: 15px;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.02));
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(0, 212, 255, 0.2);
            transition: all 0.3s;
        }
        .mtf-item:hover {
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }
        .mtf-tf { 
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem; 
            color: #888; 
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .mtf-trend { 
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem; 
            font-weight: bold; 
        }
        .mtf-trend.bullish { color: #00ff88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        .mtf-trend.bearish { color: #ff4444; text-shadow: 0 0 10px rgba(255, 68, 68, 0.5); }
        .mtf-trend.neutral { color: #888; }
        
        .progress-ring {
            width: 60px;
            height: 60px;
            margin: 0 auto;
        }
        
        /* Animated Status Dot */
        .status-dot {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: statusPulse 2s infinite;
        }
        .status-dot.online { 
            background: #00ff88; 
            box-shadow: 0 0 20px #00ff88, 0 0 40px #00ff88; 
        }
        .status-dot.offline { 
            background: #ff4444; 
            box-shadow: 0 0 20px #ff4444;
        }
        .status-dot.paused { 
            background: #ffaa00; 
            box-shadow: 0 0 20px #ffaa00;
        }
        @keyframes statusPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
        }
        
        .tf-btn {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #888;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s;
        }
        .tf-btn:hover { 
            background: rgba(0,212,255,0.2); 
            color: #fff;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.3);
        }
        .tf-btn.active { 
            background: linear-gradient(135deg, #00d4ff, #0099cc); 
            color: #000; 
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        
        /* Floating tooltip */
        .chart-tooltip {
            position: absolute;
            display: none;
            background: linear-gradient(135deg, rgba(10, 10, 30, 0.98), rgba(20, 20, 50, 0.95));
            border: 2px solid #00d4ff;
            border-radius: 12px;
            padding: 15px 20px;
            font-size: 0.85rem;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 10px 40px rgba(0, 212, 255, 0.4), 0 0 60px rgba(0, 212, 255, 0.2);
            min-width: 220px;
            backdrop-filter: blur(20px);
            transition: opacity 0.15s ease, transform 0.1s ease;
            animation: tooltipPulse 2s ease-in-out infinite;
        }
        @keyframes tooltipPulse {
            0%, 100% { box-shadow: 0 10px 40px rgba(0, 212, 255, 0.4), 0 0 60px rgba(0, 212, 255, 0.2); }
            50% { box-shadow: 0 10px 50px rgba(0, 212, 255, 0.6), 0 0 80px rgba(0, 212, 255, 0.3); }
        }
        
        /* Chart crosshair */
        .chart-crosshair-h, .chart-crosshair-v {
            position: absolute;
            pointer-events: none;
            z-index: 100;
        }
        .chart-crosshair-h {
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, transparent, #00d4ff 20%, #00d4ff 80%, transparent);
            left: 0;
        }
        .chart-crosshair-v {
            width: 1px;
            height: 100%;
            background: linear-gradient(180deg, transparent, #00d4ff 20%, #00d4ff 80%, transparent);
            top: 0;
        }
        .chart-price-label {
            position: absolute;
            right: 0;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            font-weight: bold;
            z-index: 101;
            transform: translateY(-50%);
        }
        .chart-time-label {
            position: absolute;
            bottom: 0;
            background: linear-gradient(180deg, #00d4ff, #00ff88);
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            font-weight: bold;
            z-index: 101;
            transform: translateX(-50%);
        }
        
        /* Live indicator animation */
        .live-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 8px;
            animation: livePulse 1s ease-in-out infinite;
        }
        @keyframes livePulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.7); }
            50% { opacity: 0.6; box-shadow: 0 0 0 8px rgba(0,255,136,0); }
        }
        .chart-tooltip .tt-header {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .chart-tooltip .tt-row {
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
        }
        .chart-tooltip .tt-label { color: #888; }
        .chart-tooltip .tt-value {
            color: #fff;
            font-family: 'Orbitron', sans-serif;
            font-weight: 500;
        }
        .chart-tooltip .tt-value.up { color: #00ff88; }
        .chart-tooltip .tt-value.down { color: #ff4444; }
        .chart-tooltip .tt-change {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.1);
            text-align: center;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            font-weight: bold;
        }
        
        /* Floating Refresh Button - Draggable */
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: #000;
            border: none;
            border-radius: 50%;
            cursor: grab;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 30px rgba(0, 212, 255, 0.5);
            transition: box-shadow 0.3s, transform 0.2s;
            z-index: 1000;
            user-select: none;
        }
        .refresh-btn:hover { 
            box-shadow: 0 12px 40px rgba(0, 212, 255, 0.7);
            transform: scale(1.1);
        }
        .refresh-btn:active {
            cursor: grabbing;
            transform: scale(0.95);
        }
        
        .last-update { 
            position: fixed;
            bottom: 35px;
            left: 30px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            color: #444;
            letter-spacing: 1px;
            z-index: 100;
        }
        
        /* Particle Animation Keyframes */
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        /* Neon Glow Text */
        .neon-text {
            animation: neonFlicker 1.5s infinite alternate;
        }
        @keyframes neonFlicker {
            0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {
                text-shadow: 
                    0 0 10px #00d4ff,
                    0 0 20px #00d4ff,
                    0 0 30px #00d4ff,
                    0 0 40px #00d4ff;
            }
            20%, 24%, 55% {
                text-shadow: none;
            }
        }
        
        /* AI Info Buttons */
        .info-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.05));
            border: 2px solid rgba(0, 212, 255, 0.4);
            color: #00d4ff;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-left: 10px;
            transition: all 0.3s ease;
            position: relative;
            z-index: 10;
        }
        .info-btn:hover {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.4), rgba(0, 212, 255, 0.2));
            transform: scale(1.15);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }
        .info-btn:active {
            transform: scale(0.95);
        }
        .info-btn.loading {
            animation: infoPulse 1s infinite;
        }
        @keyframes infoPulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
            50% { opacity: 0.6; box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
        }
        
        .section-title-wrapper {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 25px 0 15px 0;
        }
        .section-title-wrapper .section-title {
            margin: 0;
            flex: 1;
        }
        
        /* Market Scanner Styles */
        .market-scanner {
            background: linear-gradient(135deg, rgba(10, 15, 30, 0.9), rgba(20, 30, 60, 0.7));
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 16px;
            padding: 20px;
            margin: 25px 0;
        }
        .scanner-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }
        .scanner-card {
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2));
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .scanner-card:hover {
            transform: translateY(-3px);
            border-color: #00d4ff;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        }
        .scanner-card.active {
            border-color: #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        }
        .scanner-card.active::before {
            content: '✓';
            position: absolute;
            top: 5px;
            right: 8px;
            color: #00ff88;
            font-size: 12px;
        }
        .scanner-symbol {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            font-weight: bold;
            color: #fff;
            margin-bottom: 8px;
        }
        .scanner-price {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            color: #00d4ff;
            margin-bottom: 5px;
        }
        .scanner-change {
            font-size: 0.85rem;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        .scanner-change.up {
            color: #00ff88;
            background: rgba(0, 255, 136, 0.15);
        }
        .scanner-change.down {
            color: #ff4444;
            background: rgba(255, 68, 68, 0.15);
        }
        .scanner-stats {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.7rem;
            color: #888;
        }
        .scanner-volatility {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .volatility-bar {
            width: 40px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        .volatility-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }
        .volatility-fill.low { background: #00ff88; }
        .volatility-fill.medium { background: #ffaa00; }
        .volatility-fill.high { background: #ff4444; }
        
        /* Score badge */
        .scanner-score {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            min-width: 28px;
            text-align: center;
        }
        .scanner-score.high { background: rgba(0, 255, 136, 0.3); color: #00ff88; }
        .scanner-score.medium { background: rgba(255, 170, 0, 0.3); color: #ffaa00; }
        .scanner-score.low { background: rgba(136, 136, 136, 0.3); color: #888; }
        
        /* Signal badges */
        .signal-badge {
            font-size: 0.65rem;
            padding: 1px 4px;
            border-radius: 3px;
            margin-left: 6px;
            font-weight: bold;
            animation: signalPulse 1.5s infinite;
        }
        .signal-badge.long { background: rgba(0, 255, 136, 0.3); color: #00ff88; }
        .signal-badge.short { background: rgba(255, 68, 68, 0.3); color: #ff4444; }
        @keyframes signalPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        /* Indicator chips */
        .scanner-indicators {
            display: flex;
            gap: 4px;
            margin-top: 6px;
            flex-wrap: wrap;
        }
        .ind-chip {
            font-size: 0.6rem;
            padding: 2px 5px;
            border-radius: 3px;
            background: rgba(0, 212, 255, 0.15);
            color: #00d4ff;
        }
        .ind-chip.rsi-overbought { background: rgba(255, 68, 68, 0.2); color: #ff6666; }
        .ind-chip.rsi-oversold { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .ind-chip.rsi-neutral { background: rgba(136, 136, 136, 0.2); color: #aaa; }
        .ind-chip.trend-bullish { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .ind-chip.trend-bearish { background: rgba(255, 68, 68, 0.2); color: #ff6666; }
        
        .scanner-pulse {
            animation: scannerPulse 2s infinite;
        }
        @keyframes scannerPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .scanner-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .scanner-refresh-time {
            font-size: 0.7rem;
            color: #555;
            font-family: 'Orbitron', sans-serif;
        }
        
        .ai-analyze-btn {
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            color: #000;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .ai-analyze-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }
        .ai-analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .scanner-recommendation {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1));
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 10px;
            padding: 12px 15px;
            margin-top: 15px;
            display: none;
        }
        .scanner-recommendation.show {
            display: block;
            animation: fadeSlideIn 0.3s ease;
        }
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .recommendation-title {
            font-family: 'Orbitron', sans-serif;
            color: #00ff88;
            font-size: 0.85rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .recommendation-text {
            color: #ddd;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        /* AI Info Modal */
        .ai-modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 9999;
            animation: modalFadeIn 0.3s ease;
        }
        @keyframes modalFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .ai-modal {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            background: linear-gradient(135deg, rgba(10, 15, 30, 0.98), rgba(20, 30, 60, 0.95));
            border: 2px solid #00d4ff;
            border-radius: 20px;
            box-shadow: 
                0 20px 60px rgba(0, 212, 255, 0.3),
                0 0 100px rgba(0, 212, 255, 0.1),
                inset 0 0 60px rgba(0, 212, 255, 0.05);
            animation: modalSlideIn 0.3s ease;
            overflow: hidden;
        }
        @keyframes modalSlideIn {
            from { transform: translate(-50%, -50%) scale(0.9); opacity: 0; }
            to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
        .ai-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 25px;
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.2), transparent);
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .ai-modal-title {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ai-modal-close {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid rgba(255, 68, 68, 0.4);
            color: #ff4444;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }
        .ai-modal-close:hover {
            background: rgba(255, 68, 68, 0.4);
            transform: scale(1.1);
        }
        .ai-modal-content {
            padding: 25px;
            max-height: 60vh;
            overflow-y: auto;
            color: #ddd;
            font-size: 0.95rem;
            line-height: 1.7;
        }
        .ai-modal-content::-webkit-scrollbar {
            width: 6px;
        }
        .ai-modal-content::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        .ai-modal-content::-webkit-scrollbar-thumb {
            background: rgba(0, 212, 255, 0.3);
            border-radius: 3px;
        }
        .ai-modal-loading {
            text-align: center;
            padding: 40px;
            color: #00d4ff;
        }
        .ai-modal-loading .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(0, 212, 255, 0.2);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .ai-response {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 20px;
            border-left: 3px solid #00d4ff;
        }
        .ai-response h4 {
            color: #00ff88;
            margin-bottom: 10px;
            font-family: 'Orbitron', sans-serif;
        }
        .ai-response p {
            margin-bottom: 12px;
        }
        .ai-response ul {
            margin-left: 20px;
            margin-bottom: 12px;
        }
        .ai-response li {
            margin-bottom: 6px;
        }
        .ai-response code {
            background: rgba(0, 212, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #00d4ff;
        }
    </style>
</head>
<body>
    <!-- AI Info Modal -->
    <div class="ai-modal-overlay" id="ai-modal-overlay" onclick="closeAiModal(event)">
        <div class="ai-modal" onclick="event.stopPropagation()">
            <div class="ai-modal-header">
                <div class="ai-modal-title">
                    <span>🤖</span>
                    <span id="ai-modal-topic">AI Explanation</span>
                </div>
                <button class="ai-modal-close" onclick="closeAiModal()">×</button>
            </div>
            <div class="ai-modal-content" id="ai-modal-content">
                <div class="ai-modal-loading">
                    <div class="spinner"></div>
                    <div>AI is thinking...</div>
                </div>
            </div>
        </div>
    </div>
    
    <canvas id="bg-canvas"></canvas>
    <div class="main-content">
    <div class="container">
        <h1>🚀 JULABA TRADING SYSTEM</h1>
        
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">📡 Current Signal</h2>
                    <button class="info-btn" onclick="askAiInfo('current_signal', 'Current Signal')" title="Ask AI about this">?</button>
                </div>
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">📊 Technical Indicators</h2>
                    <button class="info-btn" onclick="askAiInfo('technical_indicators', 'Technical Indicators')" title="Ask AI about this">?</button>
                </div>
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">🎯 Market Regime</h2>
                    <button class="info-btn" onclick="askAiInfo('market_regime', 'Market Regime')" title="Ask AI about this">?</button>
                </div>
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">🛡️ Risk Manager</h2>
                    <button class="info-btn" onclick="askAiInfo('risk_manager', 'Risk Manager')" title="Ask AI about this">?</button>
                </div>
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">📈 Multi-Timeframe</h2>
                    <button class="info-btn" onclick="askAiInfo('multi_timeframe', 'Multi-Timeframe Analysis')" title="Ask AI about this">?</button>
                </div>
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">🤖 AI Filter</h2>
                    <button class="info-btn" onclick="askAiInfo('ai_filter', 'AI Filter System')" title="Ask AI about this">?</button>
                </div>
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
                <div class="section-title-wrapper">
                    <h2 class="section-title">💼 Current Position</h2>
                    <button class="info-btn" onclick="askAiInfo('current_position', 'Current Position')" title="Ask AI about this">?</button>
                </div>
                <div id="position-display">
                    <div class="signal-box neutral">
                        <div class="signal-value">NO POSITION</div>
                    </div>
                </div>
            </div>
            
            <!-- ML Model Status -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">🧠 ML Model</h2>
                    <button class="info-btn" onclick="askAiInfo('ml_model', 'Machine Learning Model')" title="Ask AI about this">?</button>
                </div>
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
                        <span class="indicator-name">Features</span>
                        <span class="indicator-value info" id="ml-features">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Last Prediction</span>
                        <span class="indicator-value" id="ml-last-pred">--</span>
                    </div>
                </div>
            </div>
            
            <!-- AI Decision Tracker -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">🎯 AI Accuracy Tracker</h2>
                    <button class="info-btn" onclick="askAiInfo('ai_tracker', 'AI Decision Tracker')" title="Ask AI about this">?</button>
                </div>
                <div id="ai-tracker-display">
                    <div style="text-align: center; padding: 8px 0;">
                        <span class="badge badge-info" id="ai-tracker-badge" style="font-size: 0.9rem; padding: 6px 14px;">TRACKING</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Total Decisions</span>
                        <span class="indicator-value" id="tracker-total">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Approval Rate</span>
                        <span class="indicator-value info" id="tracker-approval-rate">--</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Approval Accuracy</span>
                        <span class="indicator-value" id="tracker-approval-accuracy">N/A</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Net AI Value</span>
                        <span class="indicator-value" id="tracker-net-value">$0.00</span>
                    </div>
                </div>
            </div>
            
            <!-- Pre-Filter Statistics -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">🛡️ Pre-Filter Stats</h2>
                    <button class="info-btn" onclick="askAiInfo('prefilter', 'Pre-Filter System')" title="Ask AI about this">?</button>
                </div>
                <div id="prefilter-display">
                    <div style="text-align: center; padding: 8px 0;">
                        <span class="badge" id="prefilter-badge" style="font-size: 0.9rem; padding: 6px 14px;">--% PASS</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Total Signals</span>
                        <span class="indicator-value" id="prefilter-total">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Passed → AI</span>
                        <span class="indicator-value positive" id="prefilter-passed">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (Score)</span>
                        <span class="indicator-value negative" id="prefilter-score">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (ADX Low)</span>
                        <span class="indicator-value negative" id="prefilter-adx-low">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (ADX Danger)</span>
                        <span class="indicator-value negative" id="prefilter-adx-danger">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (Volume)</span>
                        <span class="indicator-value negative" id="prefilter-volume">0</span>
                    </div>
                    <div class="indicator-row">
                        <span class="indicator-name">Blocked (Confluence)</span>
                        <span class="indicator-value negative" id="prefilter-confluence">0</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="grid-3">
            <!-- Current Position -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">💼 Current Position</h2>
                    <button class="info-btn" onclick="askAiInfo('current_position', 'Current Position')" title="Ask AI about this">?</button>
                </div>
                <div id="position-display">
                    <div class="signal-box neutral">
                        <div class="signal-value">NO POSITION</div>
                    </div>
                </div>
            </div>
            
            <!-- Trading Parameters -->
            <div class="card-lg">
                <div class="section-title-wrapper">
                    <h2 class="section-title">⚙️ Trading Parameters</h2>
                    <button class="info-btn" onclick="askAiInfo('trading_parameters', 'Trading Parameters')" title="Ask AI about this">?</button>
                </div>
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
        
        <!-- Market Scanner Section -->
        <div class="market-scanner">
            <div class="scanner-header">
                <div class="section-title-wrapper" style="margin: 0; flex: 1;">
                    <h2 class="section-title">🔍 Market Scanner</h2>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="multi-pair-badge" style="display: none; background: linear-gradient(135deg, #9966ff, #00d4ff); color: #000; padding: 3px 8px; border-radius: 5px; font-size: 0.7rem; font-weight: bold; font-family: 'Orbitron', sans-serif;">MULTI-PAIR</span>
                        <span class="scanner-refresh-time" id="scanner-time">Updated: --</span>
                        <button class="ai-analyze-btn" id="ai-analyze-btn" onclick="aiAnalyzeMarkets()">
                            🤖 AI Analyze
                        </button>
                        <button class="info-btn" onclick="askAiInfo('market_scanner', 'Market Scanner')" title="Ask AI about this">?</button>
                    </div>
                </div>
            </div>
            <div class="scanner-recommendation" id="scanner-recommendation">
                <div class="recommendation-title">
                    <span>🎯</span>
                    <span>AI Recommendation</span>
                </div>
                <div class="recommendation-text" id="recommendation-text">
                    Loading AI analysis...
                </div>
            </div>
            <div class="scanner-grid" id="scanner-grid">
                <!-- Scanner cards will be populated by JavaScript -->
                <div class="scanner-card scanner-pulse">
                    <div class="scanner-symbol">Loading...</div>
                    <div class="scanner-price">$--</div>
                </div>
            </div>
        </div>
        
        <!-- Live Price Chart -->
        <div class="chart-container">
            <div class="section-title-wrapper" style="margin: 0 0 10px 0;">
                <h2 style="margin: 0;">📊 <span id="chart-symbol" style="color: #00d4ff;">--/USDT</span> <span class="live-dot"></span></h2>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span id="price-display" style="font-size: 1.4rem; color: #fff; font-family: 'Orbitron', sans-serif; animation: priceGlow 2s ease-in-out infinite; transition: transform 0.2s ease, color 0.3s ease;">$--</span>
                    <button class="info-btn" onclick="askAiInfo('live_price_chart', 'Live Price Chart')" title="Ask AI about this">?</button>
                </div>
            </div>
            <style>
                @keyframes priceGlow {
                    0%, 100% { text-shadow: 0 0 10px rgba(0,212,255,0.5); }
                    50% { text-shadow: 0 0 20px rgba(0,212,255,0.8), 0 0 30px rgba(0,255,136,0.5); }
                }
                #chart-wrapper { position: relative; overflow: hidden; border-radius: 8px; }
                #chart-hover-info span { transition: all 0.2s ease; }
            </style>
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <button class="tf-btn active" data-tf="1m" onclick="setTimeframe('1m')">1M</button>
                <button class="tf-btn" data-tf="3m" onclick="setTimeframe('3m')">3M</button>
                <button class="tf-btn" data-tf="5m" onclick="setTimeframe('5m')">5M</button>
                <button class="tf-btn" data-tf="15m" onclick="setTimeframe('15m')">15M</button>
            </div>
            <!-- Hover info panel -->
            <div id="chart-hover-info" style="display: flex; gap: 20px; margin-bottom: 8px; padding: 8px 12px; background: rgba(0,212,255,0.1); border-radius: 6px; font-size: 0.85rem; border: 1px solid rgba(0,212,255,0.3);">
                <span>📅 <span id="hover-time" style="color: #00d4ff;">--</span></span>
                <span>Open: <span id="hover-open" style="color: #fff;">--</span></span>
                <span>High: <span id="hover-high" style="color: #00ff88;">--</span></span>
                <span>Low: <span id="hover-low" style="color: #ff4444;">--</span></span>
                <span>Close: <span id="hover-close" style="color: #fff;">--</span></span>
                <span id="hover-change" style="margin-left: auto;">--</span>
            </div>
            <div id="chart-wrapper" style="position: relative;">
                <canvas id="priceChart" height="120"></canvas>
                <!-- Crosshair lines -->
                <div id="crosshair-h" class="chart-crosshair-h" style="display: none;"></div>
                <div id="crosshair-v" class="chart-crosshair-v" style="display: none;"></div>
                <div id="price-label" class="chart-price-label" style="display: none;">$--</div>
                <div id="time-label" class="chart-time-label" style="display: none;">--:--</div>
                <!-- Floating tooltip -->
                <div id="chart-float-tooltip" class="chart-tooltip">
                    <div class="tt-header"><span class="live-dot"></span>Candle Details</div>
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
            <div class="section-title-wrapper" style="margin: 0 0 10px 0;">
                <h2 style="margin: 0;">📈 Equity Curve</h2>
                <button class="info-btn" onclick="askAiInfo('equity_curve', 'Equity Curve')" title="Ask AI about this">?</button>
            </div>
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
    </div>
    </div>
    
    <!-- SYSTEM LOGS SECTION -->
    <div class="main-content" style="padding-top: 0;">
    <div class="container">
        <div id="system-logs-panel" style="
            background: linear-gradient(145deg, rgba(15, 15, 35, 0.95), rgba(5, 5, 20, 0.98));
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 16px;
            padding: 0;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(0, 212, 255, 0.1);
            overflow: hidden;
            position: relative;
        ">
            <!-- Floating Refresh Button -->
            <button id="logs-refresh-btn" onclick="fetchLogs()" style="
                position: absolute;
                top: 10px;
                right: 10px;
                width: 32px;
                height: 32px;
                background: linear-gradient(135deg, #00d4ff, #00ff88);
                color: #000;
                border: none;
                border-radius: 50%;
                cursor: grab;
                font-size: 1rem;
                font-weight: bold;
                z-index: 100;
                box-shadow: 0 3px 10px rgba(0, 212, 255, 0.5);
                transition: box-shadow 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
            " onmouseover="this.style.boxShadow='0 5px 20px rgba(0, 212, 255, 0.8)';" onmouseout="this.style.boxShadow='0 3px 10px rgba(0, 212, 255, 0.5)';">⟳</button>
            
            <!-- Log Header -->
            <div style="
                background: linear-gradient(90deg, rgba(0, 212, 255, 0.2), rgba(0, 255, 136, 0.1));
                padding: 12px 25px;
                border-bottom: 1px solid rgba(0, 212, 255, 0.3);
            ">
                <h2 style="
                    font-family: 'Orbitron', sans-serif;
                    color: #00d4ff;
                    font-size: 1rem;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 3px;
                    text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
                ">📜 SYSTEM LOGS</h2>
            </div>
            
            <!-- Log Content Area -->
            <div id="logs-container" style="
                height: 400px;
                overflow-y: auto;
                overflow-x: auto;
                padding: 20px 25px;
                background: rgba(0, 0, 0, 0.4);
            ">
                <pre id="logs-content" style="
                    font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
                    font-size: 0.9rem;
                    line-height: 2;
                    margin: 0;
                    color: #aaa;
                    white-space: pre-wrap;
                    word-break: break-word;
                ">Loading logs...</pre>
            </div>
        </div>
    </div>
    </div>
    
    <!-- Draggable Refresh Button Script -->
    <script>
        (function() {
            var btn = document.getElementById('logs-refresh-btn');
            let isDragging = false;
            let startX, startY, startLeft, startTop;
            
            btn.addEventListener('mousedown', function(e) {
                isDragging = true;
                btn.style.cursor = 'grabbing';
                startX = e.clientX;
                startY = e.clientY;
                var rect = btn.getBoundingClientRect();
                var parent = btn.parentElement.getBoundingClientRect();
                startLeft = rect.left - parent.left;
                startTop = rect.top - parent.top;
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                var dx = e.clientX - startX;
                var dy = e.clientY - startY;
                btn.style.left = (startLeft + dx) + 'px';
                btn.style.top = (startTop + dy) + 'px';
                btn.style.right = 'auto';
            });
            
            document.addEventListener('mouseup', function() {
                if (isDragging) {
                    isDragging = false;
                    btn.style.cursor = 'grab';
                }
            });
        })();
    </script>
    
    <div class="last-update">Last update: <span id="last-update">--</span> | Auto-refresh: <span id="auto-refresh-status" style="color:#00ff88;">ON</span></div>
    <button id="main-refresh-btn" class="refresh-btn" onclick="refreshAll()" title="Drag to move, Click to refresh">⚡</button>
    
    <!-- Draggable Main Refresh Button Script -->
    <script>
        (function() {
            var btn = document.getElementById('main-refresh-btn');
            let isDragging = false;
            let hasMoved = false;
            let startX, startY, startLeft, startBottom, startRight;
            
            btn.addEventListener('mousedown', function(e) {
                isDragging = true;
                hasMoved = false;
                btn.style.cursor = 'grabbing';
                startX = e.clientX;
                startY = e.clientY;
                var rect = btn.getBoundingClientRect();
                startRight = window.innerWidth - rect.right;
                startBottom = window.innerHeight - rect.bottom;
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                var dx = e.clientX - startX;
                var dy = e.clientY - startY;
                if (Math.abs(dx) > 3 || Math.abs(dy) > 3) hasMoved = true;
                btn.style.right = Math.max(10, startRight - dx) + 'px';
                btn.style.bottom = Math.max(10, startBottom + dy) + 'px';
            });
            
            document.addEventListener('mouseup', function() {
                if (isDragging) {
                    isDragging = false;
                    btn.style.cursor = 'grab';
                }
            });
            
            // Prevent click from firing after drag
            btn.addEventListener('click', function(e) {
                if (hasMoved) {
                    e.stopPropagation();
                    e.preventDefault();
                }
            }, true);
        })();
    </script>
    
    <!-- 3D Animated Background -->
    <script>
        // Three.js Particle Background
        var canvas = document.getElementById('bg-canvas');
        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        var renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        // Create particles
        var particlesGeometry = new THREE.BufferGeometry();
        var particleCount = 2000;
        var posArray = new Float32Array(particleCount * 3);
        var colorsArray = new Float32Array(particleCount * 3);
        
        for(let i = 0; i < particleCount * 3; i += 3) {
            posArray[i] = (Math.random() - 0.5) * 50;
            posArray[i + 1] = (Math.random() - 0.5) * 50;
            posArray[i + 2] = (Math.random() - 0.5) * 50;
            
            // Cyan to green gradient colors
            var t = Math.random();
            colorsArray[i] = t * 0 + (1-t) * 0;        // R
            colorsArray[i + 1] = t * 1 + (1-t) * 0.83; // G
            colorsArray[i + 2] = t * 0.53 + (1-t) * 1; // B
        }
        
        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
        
        var particlesMaterial = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });
        
        var particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        scene.add(particlesMesh);
        
        // Add connecting lines
        var linesGeometry = new THREE.BufferGeometry();
        var linePositions = [];
        for(let i = 0; i < 100; i++) {
            var x1 = (Math.random() - 0.5) * 30;
            var y1 = (Math.random() - 0.5) * 30;
            var z1 = (Math.random() - 0.5) * 30;
            var x2 = x1 + (Math.random() - 0.5) * 5;
            var y2 = y1 + (Math.random() - 0.5) * 5;
            var z2 = z1 + (Math.random() - 0.5) * 5;
            linePositions.push(x1, y1, z1, x2, y2, z2);
        }
        linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        var linesMaterial = new THREE.LineBasicMaterial({ 
            color: 0x00d4ff, 
            transparent: true, 
            opacity: 0.15 
        });
        var lines = new THREE.LineSegments(linesGeometry, linesMaterial);
        scene.add(lines);
        
        camera.position.z = 15;
        
        // Mouse interaction
        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', function(e) {
            mouseX = (e.clientX / window.innerWidth) * 2 - 1;
            mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
        });
        
        // Animation loop
        function animateBg() {
            requestAnimationFrame(animateBg);
            
            particlesMesh.rotation.x += 0.0003;
            particlesMesh.rotation.y += 0.0005;
            lines.rotation.x += 0.0002;
            lines.rotation.y += 0.0003;
            
            // Follow mouse
            particlesMesh.rotation.x += mouseY * 0.0005;
            particlesMesh.rotation.y += mouseX * 0.0005;
            
            renderer.render(scene, camera);
        }
        animateBg();
        
        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
    
    <script>
        console.log('=== JULABA DASHBOARD SCRIPT STARTING ===');
        let equityChart = null;
        let priceChart = null;
        let currentTimeframe = '1m';
        var autoRefresh = true;
        
        // AI Info Modal Functions
        function askAiInfo(topic, displayName) {
            var modal = document.getElementById('ai-modal-overlay');
            var topicEl = document.getElementById('ai-modal-topic');
            var contentEl = document.getElementById('ai-modal-content');
            
            if (topicEl) topicEl.textContent = displayName;
            if (contentEl) contentEl.innerHTML = '<div class="ai-modal-loading"><div class="spinner"></div><div>AI is analyzing ' + displayName + '...</div></div>';
            if (modal) modal.style.display = 'block';
            
            // Fetch AI explanation
            fetch('/api/ai-explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic: topic, display_name: displayName })
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.error) {
                    contentEl.innerHTML = '<div class="ai-response"><p style="color: #ff4444;">❌ ' + data.error + '</p></div>';
                } else {
                    contentEl.innerHTML = '<div class="ai-response">' + formatAiResponse(data.explanation) + '</div>';
                }
            })
            .catch(function(e) {
                contentEl.innerHTML = '<div class="ai-response"><p style="color: #ff4444;">❌ Error: ' + e.message + '</p></div>';
            });
        }
        
        function formatAiResponse(text) {
            // Convert markdown-like formatting to HTML
            var html = text
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                .replace(/`(.+?)`/g, '<code>$1</code>')
                .replace(/^### (.+)$/gm, '<h4>$1</h4>')
                .replace(/^## (.+)$/gm, '<h4>$1</h4>')
                .replace(/^# (.+)$/gm, '<h4>$1</h4>')
                .replace(/^[\\-\\*] (.+)$/gm, '<li>$1</li>')
                .replace(/^• (.+)$/gm, '<li>$1</li>')
                .replace(/\\n\\n/g, '</p><p>')
                .replace(/\\n/g, '<br>');
            
            // Wrap lists
            html = html.replace(/(<li>.*?<\\/li>)+/gs, '<ul>$&</ul>');
            
            return '<p>' + html + '</p>';
        }
        
        function closeAiModal(event) {
            if (event && event.target !== document.getElementById('ai-modal-overlay')) {
                return;
            }
            document.getElementById('ai-modal-overlay').style.display = 'none';
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeAiModal();
        });
        
        // Market Scanner Variables
        var currentSymbol = '';
        var marketScanData = [];
        
        function fetchMarketScan() {
            fetch('/api/market-scan')
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.pairs) {
                        marketScanData = data.pairs;
                        currentSymbol = data.current_symbol || '';
                        updateMarketScanner(data.pairs, data.current_symbol);
                    }
                    // Show multi-pair badge if enabled
                    var mpBadge = document.getElementById('multi-pair-badge');
                    if (mpBadge) {
                        mpBadge.style.display = data.multi_pair_enabled ? 'inline-block' : 'none';
                        if (data.multi_pair_count) mpBadge.textContent = 'MULTI (' + data.multi_pair_count + ')';
                    }
                    var scanTime = document.getElementById('scanner-time');
                    if (scanTime) scanTime.textContent = 'Updated: ' + new Date().toLocaleTimeString();
                })
                .catch(function(e) { console.error('Market scan error:', e); });
        }
        
        function updateMarketScanner(pairs, activeSymbol) {
            var grid = document.getElementById('scanner-grid');
            if (!grid || !pairs) return;
            
            // Sort pairs: active symbol always first, then by score
            var sortedPairs = pairs.slice().sort(function(a, b) {
                if (a.symbol === activeSymbol) return -1;
                if (b.symbol === activeSymbol) return 1;
                return (b.score || 0) - (a.score || 0);
            });
            
            var html = '';
            sortedPairs.forEach(function(pair, idx) {
                var isActive = pair.symbol === activeSymbol;
                var changeClass = pair.change >= 0 ? 'up' : 'down';
                var changeSign = pair.change >= 0 ? '+' : '';
                var volPct = Math.min(100, (pair.volatility || 0) * 25); // Scale volatility
                var volClass = volPct < 33 ? 'low' : volPct < 66 ? 'medium' : 'high';
                var score = pair.score || 0;
                var scoreClass = score >= 70 ? 'high' : score >= 45 ? 'medium' : 'low';
                var signalBadge = '';
                if (pair.signal === 1) signalBadge = '<span class="signal-badge long">LONG</span>';
                else if (pair.signal === -1) signalBadge = '<span class="signal-badge short">SHORT</span>';
                
                html += '<div class="scanner-card' + (isActive ? ' active' : '') + '" onclick="switchToSymbol(\\'' + pair.symbol + '\\')" title="Click to switch">';
                
                // Top row: symbol + score
                html += '<div class="scanner-header">';
                html += '<div class="scanner-symbol">' + pair.symbol.replace('USDT', '') + '</div>';
                html += '<div class="scanner-score ' + scoreClass + '">' + Math.round(score) + '</div>';
                html += '</div>';
                
                // Price and change
                html += '<div class="scanner-price">$' + formatPrice(pair.price) + '</div>';
                html += '<div class="scanner-change ' + changeClass + '">' + changeSign + pair.change.toFixed(2) + '%' + signalBadge + '</div>';
                
                // Indicators row
                if (pair.rsi !== undefined) {
                    var rsiClass = pair.rsi > 70 ? 'overbought' : pair.rsi < 30 ? 'oversold' : 'neutral';
                    html += '<div class="scanner-indicators">';
                    html += '<span class="ind-chip rsi-' + rsiClass + '">RSI ' + Math.round(pair.rsi) + '</span>';
                    html += '<span class="ind-chip">ADX ' + Math.round(pair.adx || 0) + '</span>';
                    html += '<span class="ind-chip trend-' + pair.trend + '">' + (pair.trend === 'bullish' ? '↑' : '↓') + '</span>';
                    html += '</div>';
                }
                
                // Volume bar
                html += '<div class="scanner-stats">';
                html += '<div class="scanner-volatility">';
                html += '<span>Vol:</span>';
                html += '<div class="volatility-bar"><div class="volatility-fill ' + volClass + '" style="width: ' + volPct + '%"></div></div>';
                html += '</div>';
                html += '</div>';
                html += '</div>';
            });
            
            grid.innerHTML = html;
        }
        
        function formatPrice(price) {
            if (price >= 1000) return price.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
            if (price >= 1) return price.toFixed(2);
            if (price >= 0.01) return price.toFixed(4);
            return price.toFixed(6);
        }
        
        function formatVolume(vol) {
            if (vol >= 1e9) return (vol / 1e9).toFixed(1) + 'B';
            if (vol >= 1e6) return (vol / 1e6).toFixed(1) + 'M';
            if (vol >= 1e3) return (vol / 1e3).toFixed(0) + 'K';
            return vol.toFixed(0);
        }
        
        function switchToSymbol(symbol) {
            if (symbol === currentSymbol) return;
            
            // Show confirmation via AI
            var confirmSwitch = confirm('Switch trading to ' + symbol + '?');
            if (!confirmSwitch) return;
            
            fetch('/api/switch-symbol', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol })
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    currentSymbol = symbol;
                    fetchMarketScan();
                    fetchData();
                    fetchPriceData();
                    alert('Switched to ' + symbol);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(function(e) { alert('Switch error: ' + e.message); });
        }
        
        function aiAnalyzeMarkets() {
            var btn = document.getElementById('ai-analyze-btn');
            var recDiv = document.getElementById('scanner-recommendation');
            var recText = document.getElementById('recommendation-text');
            
            if (btn) btn.disabled = true;
            if (recDiv) recDiv.classList.add('show');
            if (recText) recText.innerHTML = '<span class="scanner-pulse">🤖 AI is analyzing all markets...</span>';
            
            fetch('/api/ai-analyze-markets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (btn) btn.disabled = false;
                if (data.recommendation) {
                    var cacheInfo = '';
                    if (data.cached) {
                        var mins = Math.floor(data.cache_expires_in / 60);
                        var secs = data.cache_expires_in % 60;
                        cacheInfo = '<div style="color:#888;font-size:0.75em;margin-top:8px;">📦 Cached result • refreshes in ' + mins + 'm ' + secs + 's</div>';
                    } else if (data.cache_expires_in) {
                        cacheInfo = '<div style="color:#888;font-size:0.75em;margin-top:8px;">✨ Fresh analysis • valid for 5 minutes</div>';
                    }
                    recText.innerHTML = formatAiResponse(data.recommendation) + cacheInfo;
                } else if (data.error) {
                    recText.innerHTML = '<span style="color:#ff4444;">❌ ' + data.error + '</span>';
                }
            })
            .catch(function(e) {
                if (btn) btn.disabled = false;
                recText.innerHTML = '<span style="color:#ff4444;">❌ Error: ' + e.message + '</span>';
            });
        }
        
        function fetchData() {
            fetch('/api/data')
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    updateDashboard(data);
                    var lastUpdateEl = document.getElementById('last-update');
                    if (lastUpdateEl) lastUpdateEl.textContent = new Date().toLocaleTimeString();
                })
                .catch(function(e) { console.error('Fetch error:', e); });
        }
        
        function fetchPriceData() {
            fetch('/api/ohlc?tf=' + currentTimeframe)
                .then(function(response) { return response.json(); })
                .then(function(data) { updatePriceChart(data); })
                .catch(function(e) { console.error('Price fetch error:', e); });
        }
        
        function fetchLogs() {
            fetch('/api/logs?count=150')
                .then(function(response) { return response.json(); })
                .then(function(data) { updateLogs(data.logs || []); })
                .catch(function(e) {
                    console.error('Logs fetch error:', e);
                    var logsEl = document.getElementById('logs-content');
                    if (logsEl) logsEl.innerHTML = '<span style="color:#ff4444;">Error loading logs</span>';
                });
        }
        
        function updateLogs(logs) {
            var container = document.getElementById('logs-content');
            if (!logs || logs.length === 0) {
                container.textContent = 'No logs available';
                container.style.color = '#666';
                return;
            }
            
            // Filter out noisy log messages
            var filteredLogs = logs.filter(function(log) {
                var msg = log.message || '';
                if (msg.includes('httpx') || msg.includes('HTTP Request:')) return false;
                if (msg.includes('getUpdates') || msg.includes('telegram.org/bot')) return false;
                if (msg.includes('Signal confirmed with confluence')) return false;
                if (msg.includes('Heartbeat') || msg.includes('heartbeat')) return false;
                return true;
            });
            
            if (filteredLogs.length === 0) {
                container.textContent = 'No significant logs (filtered noise)';
                container.style.color = '#666';
                return;
            }
            
            // Build log entries as formatted text
            var logHtml = '';
            filteredLogs.forEach(function(log) {
                var levelColor = '#888';
                var levelBg = 'rgba(136,136,136,0.2)';
                if (log.level === 'ERROR') { levelColor = '#ff4444'; levelBg = 'rgba(255,68,68,0.3)'; }
                else if (log.level === 'WARNING') { levelColor = '#ffaa00'; levelBg = 'rgba(255,170,0,0.3)'; }
                else if (log.level === 'INFO') { levelColor = '#00d4ff'; levelBg = 'rgba(0,212,255,0.2)'; }
                
                var msgColor = levelColor;
                var msg = log.message || '';
                if (msg.includes('✅')) msgColor = '#00ff88';
                else if (msg.includes('❌') || msg.includes('🚫')) msgColor = '#ff4444';
                else if (msg.includes('🧠')) msgColor = '#9966ff';
                else if (msg.includes('📈') || msg.includes('📊')) msgColor = '#00d4ff';
                
                // Short level indicator
                var levelShort = log.level === 'WARNING' ? 'WARN' : log.level === 'ERROR' ? 'ERR' : 'INF';
                var timeShort = (log.time.split(' ')[1] || log.time).substring(0, 5);
                
                logHtml += '<div style="display:flex; align-items:flex-start; gap:8px; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.03);">' +
                    '<span style="color:#555; font-size:0.7rem; flex-shrink:0;">' + timeShort + '</span>' +
                    '<span style="background:' + levelBg + '; color:' + levelColor + '; padding:1px 5px; border-radius:3px; font-size:0.65rem; flex-shrink:0; font-weight:bold;">' + levelShort + '</span>' +
                    '<span style="color:' + msgColor + '; flex:1; word-break:break-word;">' + msg + '</span>' +
                '</div>';
            });
            
            container.innerHTML = logHtml;
            
            // Scroll to bottom
            var logsContainer = document.getElementById('logs-container');
            logsContainer.scrollTop = logsContainer.scrollHeight;
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
            fetchMarketScan();
        }
        
        function updatePriceChart(data) {
            if (!data || !data.candles || data.candles.length === 0) {
                console.log('No candle data received:', data);
                return;
            }
            
            var candles = data.candles;
            var ctx = document.getElementById('priceChart').getContext('2d');
            
            // Update current price display
            var lastCandle = candles[candles.length - 1];
            var firstCandle = candles[0];
            var priceEl = document.getElementById('price-display');
            priceEl.textContent = '$' + lastCandle.c.toFixed(4);
            var priceChange = ((lastCandle.c - firstCandle.o) / firstCandle.o * 100).toFixed(2);
            priceEl.style.color = lastCandle.c >= firstCandle.o ? '#00ff88' : '#ff4444';
            
            // Update high/low/volume
            var highs = candles.map(c => c.h);
            var lows = candles.map(c => c.l);
            var volumes = candles.map(c => c.v);
            document.getElementById('chart-high').textContent = '$' + Math.max(...highs).toFixed(4);
            document.getElementById('chart-low').textContent = '$' + Math.min(...lows).toFixed(4);
            document.getElementById('chart-volume').textContent = volumes.reduce((a,b) => a+b, 0).toLocaleString();
            
            // Prepare candlestick data - convert timestamp to Date object
            var ohlcData = candles.map(c => ({
                x: new Date(c.t),
                o: c.o,
                h: c.h,
                l: c.l,
                c: c.c
            }));
            
            // Determine trend for beautiful gradient colors
            var isBullish = lastCandle.c >= firstCandle.o;
            var mainColor = isBullish ? '#00ff88' : '#ff4444';
            
            // Prepare smooth line data (close prices) for beautiful chart
            var closeData = candles.map(c => ({ x: new Date(c.t), y: c.c }));
            var highData = candles.map(c => ({ x: new Date(c.t), y: c.h }));
            var lowData = candles.map(c => ({ x: new Date(c.t), y: c.l }));
            
            // Create gradient function
            function createGradient(ctx, color) {
                var gradient = ctx.createLinearGradient(0, 0, 0, 180);
                if (color === '#00ff88') {
                    gradient.addColorStop(0, 'rgba(0, 255, 136, 0.5)');
                    gradient.addColorStop(0.3, 'rgba(0, 255, 136, 0.2)');
                    gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
                } else {
                    gradient.addColorStop(0, 'rgba(255, 68, 68, 0.5)');
                    gradient.addColorStop(0.3, 'rgba(255, 68, 68, 0.2)');
                    gradient.addColorStop(1, 'rgba(255, 68, 68, 0)');
                }
                return gradient;
            }
            
            if (priceChart) {
                // Update with smooth animation
                priceChart.data.datasets[0].data = highData;
                priceChart.data.datasets[1].data = lowData;
                priceChart.data.datasets[2].data = closeData;
                priceChart.data.datasets[2].borderColor = mainColor;
                priceChart.data.datasets[2].pointBackgroundColor = mainColor;
                priceChart.data.datasets[2].backgroundColor = createGradient(ctx, mainColor);
                priceChart.update('default');
                
                // Pulse effect on new data
                priceEl.style.transform = 'scale(1.15)';
                priceEl.style.textShadow = '0 0 30px ' + mainColor + ', 0 0 60px ' + mainColor;
                setTimeout(() => { 
                    priceEl.style.transform = 'scale(1)'; 
                    priceEl.style.textShadow = '';
                }, 300);
            } else {
                priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [
                            // High range (subtle glow)
                            {
                                label: 'High',
                                data: highData,
                                borderColor: 'rgba(0, 255, 136, 0.15)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.4
                            },
                            // Low range (fill area between high and low)
                            {
                                label: 'Low',
                                data: lowData,
                                borderColor: 'rgba(255, 68, 68, 0.15)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: '-1',
                                backgroundColor: 'rgba(0, 212, 255, 0.03)',
                                tension: 0.4
                            },
                            // Main price line with glow effect
                            {
                                label: 'Price',
                                data: closeData,
                                borderColor: mainColor,
                                borderWidth: 3,
                                pointRadius: 0,
                                pointHoverRadius: 8,
                                pointHoverBackgroundColor: mainColor,
                                pointHoverBorderColor: '#fff',
                                pointHoverBorderWidth: 3,
                                fill: true,
                                backgroundColor: createGradient(ctx, mainColor),
                                tension: 0.4,
                                cubicInterpolationMode: 'monotone'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        animation: {
                            duration: 800,
                            easing: 'easeInOutQuart'
                        },
                        interaction: {
                            mode: 'index',
                            intersect: false
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: { enabled: false }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: { unit: 'minute', displayFormats: { minute: 'HH:mm' } },
                                grid: { 
                                    color: 'rgba(0, 212, 255, 0.08)',
                                    drawBorder: false
                                },
                                ticks: { 
                                    color: '#555', 
                                    maxTicksLimit: 8,
                                    font: { family: 'Orbitron', size: 9 }
                                }
                            },
                            y: {
                                position: 'right',
                                grid: { 
                                    color: 'rgba(0, 212, 255, 0.08)',
                                    drawBorder: false
                                },
                                ticks: { 
                                    color: '#555', 
                                    callback: v => '$' + v.toFixed(2),
                                    font: { family: 'Orbitron', size: 9 }
                                }
                            }
                        },
                        onHover: function(event, elements) {
                            if (elements.length > 0) {
                                var dataIndex = elements[0].index;
                                var candle = candles[dataIndex];
                                if (candle) {
                                    var date = new Date(candle.t);
                                    document.getElementById('hover-time').textContent = date.toLocaleTimeString();
                                    document.getElementById('hover-open').textContent = '$' + candle.o.toFixed(4);
                                    document.getElementById('hover-high').textContent = '$' + candle.h.toFixed(4);
                                    document.getElementById('hover-low').textContent = '$' + candle.l.toFixed(4);
                                    document.getElementById('hover-close').textContent = '$' + candle.c.toFixed(4);
                                    var change = ((candle.c - candle.o) / candle.o * 100);
                                    var changeEl = document.getElementById('hover-change');
                                    changeEl.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                                    changeEl.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                                }
                            }
                        }
                    }
                });
                
                // Add mouse/touch move listener for crosshair and tooltip
                var canvas = document.getElementById('priceChart');
                var chartWrapper = document.getElementById('chart-wrapper');
                var floatTooltip = document.getElementById('chart-float-tooltip');
                var crosshairH = document.getElementById('crosshair-h');
                var crosshairV = document.getElementById('crosshair-v');
                var priceLabel = document.getElementById('price-label');
                var timeLabel = document.getElementById('time-label');
                
                function handleChartInteraction(e) {
                    var rect = canvas.getBoundingClientRect();
                    let clientX, clientY;
                    
                    // Handle both mouse and touch events
                    if (e.touches && e.touches.length > 0) {
                        clientX = e.touches[0].clientX;
                        clientY = e.touches[0].clientY;
                    } else {
                        clientX = e.clientX;
                        clientY = e.clientY;
                    }
                    
                    var x = clientX - rect.left;
                    var y = clientY - rect.top;
                    
                    // Show crosshair
                    crosshairH.style.display = 'block';
                    crosshairH.style.top = y + 'px';
                    crosshairV.style.display = 'block';
                    crosshairV.style.left = x + 'px';
                    
                    var elements = priceChart.getElementsAtEventForMode(e, 'index', { intersect: false }, false);
                    
                    if (elements.length > 0) {
                        var dataIndex = elements[0].index;
                        var dataset = priceChart.data.datasets[0].data;
                        if (dataset && dataset[dataIndex]) {
                            var d = dataset[dataIndex];
                            var date = new Date(d.x);
                            var change = ((d.c - d.o) / d.o * 100);
                            
                            // Show price label on crosshair
                            priceLabel.style.display = 'block';
                            priceLabel.style.top = y + 'px';
                            priceLabel.textContent = '$' + d.c.toFixed(4);
                            
                            // Show time label on crosshair
                            timeLabel.style.display = 'block';
                            timeLabel.style.left = x + 'px';
                            timeLabel.textContent = date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                            
                            // Update header info bar with animation
                            document.getElementById('hover-time').textContent = date.toLocaleTimeString();
                            document.getElementById('hover-open').textContent = '$' + d.o.toFixed(4);
                            document.getElementById('hover-high').textContent = '$' + d.h.toFixed(4);
                            document.getElementById('hover-low').textContent = '$' + d.l.toFixed(4);
                            document.getElementById('hover-close').textContent = '$' + d.c.toFixed(4);
                            var changeEl = document.getElementById('hover-change');
                            changeEl.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                            changeEl.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                            
                            // Update floating tooltip
                            document.getElementById('tt-time').textContent = date.toLocaleString();
                            document.getElementById('tt-open').textContent = '$' + d.o.toFixed(4);
                            document.getElementById('tt-high').textContent = '$' + d.h.toFixed(4);
                            document.getElementById('tt-low').textContent = '$' + d.l.toFixed(4);
                            document.getElementById('tt-close').textContent = '$' + d.c.toFixed(4);
                            document.getElementById('tt-close').className = 'tt-value ' + (d.c >= d.o ? 'up' : 'down');
                            var ttChange = document.getElementById('tt-change');
                            ttChange.textContent = (change >= 0 ? '▲ +' : '▼ ') + change.toFixed(2) + '%';
                            ttChange.style.color = change >= 0 ? '#00ff88' : '#ff4444';
                            
                            // Position tooltip near cursor with smooth animation
                            let tooltipX = x + 20;
                            let tooltipY = y - 100;
                            
                            // Keep tooltip within canvas bounds
                            if (tooltipX + 220 > rect.width) {
                                tooltipX = x - 240;
                            }
                            if (tooltipY < 10) {
                                tooltipY = y + 20;
                            }
                            if (tooltipY + 180 > rect.height) {
                                tooltipY = rect.height - 190;
                            }
                            
                            floatTooltip.style.left = tooltipX + 'px';
                            floatTooltip.style.top = tooltipY + 'px';
                            floatTooltip.style.display = 'block';
                            floatTooltip.style.opacity = '1';
                        }
                    }
                }
                
                function hideChartElements() {
                    floatTooltip.style.display = 'none';
                    crosshairH.style.display = 'none';
                    crosshairV.style.display = 'none';
                    priceLabel.style.display = 'none';
                    timeLabel.style.display = 'none';
                }
                
                // Mouse events
                canvas.addEventListener('mousemove', handleChartInteraction);
                canvas.addEventListener('mouseleave', hideChartElements);
                
                // Touch events for mobile
                canvas.addEventListener('touchmove', function(e) {
                    e.preventDefault();
                    handleChartInteraction(e);
                }, { passive: false });
                canvas.addEventListener('touchstart', function(e) {
                    handleChartInteraction(e);
                });
                canvas.addEventListener('touchend', hideChartElements);
            }
        }
        
        function updateDashboard(data) {
            console.log('updateDashboard called with data:', data ? 'yes' : 'no');
            try {
                // Helper function to safely set text
                function setText(id, value) {
                    var el = document.getElementById(id);
                    if (el) el.textContent = value;
                }
                
                function setClass(id, className) {
                    var el = document.getElementById(id);
                    if (el) el.className = className;
                }
                
                // Status
                if (data.status) {
                    var statusDot = document.getElementById('status-dot');
                    var statusText = document.getElementById('status-text');
                    if (data.status.connected) {
                        if (data.status.paused) {
                            if (statusDot) statusDot.className = 'status-dot paused';
                            if (statusText) statusText.textContent = 'PAUSED';
                        } else {
                            if (statusDot) statusDot.className = 'status-dot online';
                            if (statusText) statusText.textContent = 'RUNNING';
                        }
                    } else {
                        if (statusDot) statusDot.className = 'status-dot offline';
                        if (statusText) statusText.textContent = 'OFFLINE';
                    }
                    // Update chart symbol
                    var chartSymbol = document.getElementById('chart-symbol');
                    if (chartSymbol && data.status.symbol) {
                        // Format symbol nicely (LINKUSDT -> LINK/USDT)
                        var sym = data.status.symbol;
                        if (sym.indexOf('/') === -1 && sym.endsWith('USDT')) {
                            sym = sym.replace('USDT', '/USDT');
                        }
                        chartSymbol.textContent = sym;
                        // Keep global currentSymbol in sync
                        currentSymbol = data.status.symbol;
                    }
                }
                
                // Balance
                var bal = data.balance || {};
                setText('balance', '$' + (bal.current || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}));
                var changePct = bal.change_pct || 0;
                var changeEl = document.getElementById('balance-change');
                if (changeEl) {
                    changeEl.textContent = (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%';
                    changeEl.style.color = changePct >= 0 ? '#00ff88' : '#ff4444';
                }
                
                // P&L
                var pnl = data.pnl || {};
                var todayPnl = pnl.today || 0;
                var todayEl = document.getElementById('today-pnl');
                if (todayEl) {
                    todayEl.textContent = (todayPnl >= 0 ? '+$' : '-$') + Math.abs(todayPnl).toFixed(2);
                    todayEl.className = 'value ' + (todayPnl >= 0 ? 'positive' : 'negative');
                }
                
                var totalPnl = pnl.total || 0;
                var totalEl = document.getElementById('total-pnl');
                if (totalEl) {
                    totalEl.textContent = (totalPnl >= 0 ? '+$' : '-$') + Math.abs(totalPnl).toFixed(2);
                    totalEl.className = 'value ' + (totalPnl >= 0 ? 'positive' : 'negative');
                }
                
                // Win Rate
                setText('win-rate', ((pnl.win_rate || 0) * 100).toFixed(1) + '%');
                setText('win-loss', (pnl.winning || 0) + 'W / ' + ((pnl.trades || 0) - (pnl.winning || 0)) + 'L');
                setText('total-trades', pnl.trades || 0);
                
                // Current Signal
            var signal = data.current_signal || {};
                var sigBox = document.querySelector('#signal-display .signal-box');
                var sigValue = document.querySelector('#signal-display .signal-value');
                if (sigBox && sigValue) {
                    if (signal.direction) {
                        sigBox.className = 'signal-box ' + (signal.direction === 'LONG' ? 'long' : signal.direction === 'SHORT' ? 'short' : 'neutral');
                        sigValue.textContent = signal.direction;
                        setText('sig-confidence', ((signal.confidence || 0) * 100).toFixed(0) + '%');
                        setText('sig-entry', signal.entry ? '$' + signal.entry.toFixed(4) : '--');
                        setText('sig-sl', signal.stop_loss ? '$' + signal.stop_loss.toFixed(4) : '--');
                        setText('sig-tp', signal.take_profit ? '$' + signal.take_profit.toFixed(4) : '--');
                    } else {
                        sigBox.className = 'signal-box neutral';
                        sigValue.textContent = 'WAITING';
                    }
                }
                
                // Technical Indicators
                var ind = data.indicators || {};
                setText('ind-rsi', ind.rsi ? ind.rsi.toFixed(1) : '--');
                if (ind.rsi) {
                    var rsiBar = document.getElementById('rsi-bar');
                    if (rsiBar) {
                        rsiBar.style.width = ind.rsi + '%';
                        rsiBar.style.background = ind.rsi > 70 ? '#ff4444' : ind.rsi < 30 ? '#00ff88' : '#00d4ff';
                    }
                }
                setText('ind-macd', ind.macd_signal || '--');
                setText('ind-adx', ind.adx ? ind.adx.toFixed(1) : '--');
                setText('ind-atr', ind.atr ? '$' + ind.atr.toFixed(4) : '--');
                setText('ind-bb', ind.bb_position || '--');
                setText('ind-volume', ind.volume_ratio ? ind.volume_ratio.toFixed(2) + 'x' : '--');
                
                // Market Regime
                var regime = data.regime || {};
                var regimeBadge = document.getElementById('regime-badge');
                if (regimeBadge) {
                    regimeBadge.textContent = regime.regime || 'UNKNOWN';
                    regimeBadge.className = 'badge ' + (regime.regime === 'TRENDING' || regime.regime === 'STRONG_TRENDING' ? 'badge-success' : regime.regime === 'CHOPPY' ? 'badge-danger' : 'badge-info');
                }
                setText('regime-adx', regime.adx ? regime.adx.toFixed(1) : '--');
                setText('regime-hurst', regime.hurst ? regime.hurst.toFixed(3) : '--');
                setText('regime-volatility', regime.volatility || '--');
                var tradeableEl = document.getElementById('regime-tradeable');
                if (tradeableEl) {
                    tradeableEl.textContent = regime.tradeable ? '✅ YES' : '❌ NO';
                    tradeableEl.className = 'indicator-value ' + (regime.tradeable ? 'positive' : 'negative');
                }
                
                // Risk Manager
                var risk = data.risk || {};
                var canTradeEl = document.getElementById('risk-can-trade');
                if (canTradeEl) {
                    canTradeEl.textContent = risk.can_trade ? '✅ YES' : '🛑 NO';
                    canTradeEl.className = 'indicator-value ' + (risk.can_trade ? 'positive' : 'negative');
                }
                setText('risk-mode', risk.dd_mode || 'NORMAL');
                setText('risk-base', ((risk.base_risk || 0.02) * 100).toFixed(1) + '%');
                setText('risk-adjusted', ((risk.adjusted_risk || 0.02) * 100).toFixed(2) + '%');
                setText('risk-kelly', ((risk.kelly_risk || 0.02) * 100).toFixed(2) + '%');
                var dailyPnl = risk.daily_pnl || 0;
                var dailyEl = document.getElementById('risk-daily');
                if (dailyEl) {
                    dailyEl.textContent = (dailyPnl >= 0 ? '+$' : '-$') + Math.abs(dailyPnl).toFixed(2);
                    dailyEl.className = 'indicator-value ' + (dailyPnl >= 0 ? 'positive' : 'negative');
                }
                
                // MTF Analysis
                var mtf = data.mtf || {};
                var mtfPrimary = mtf.primary || {};
                var mtfSecondary = mtf.secondary || {};
                var mtfHigher = mtf.higher || {};
                updateMTF('mtf-3m', (mtfPrimary.trend && mtfPrimary.trend.direction) || '--');
                updateMTF('mtf-15m', (mtfSecondary.trend && mtfSecondary.trend.direction) || '--');
                updateMTF('mtf-1h', (mtfHigher.trend && mtfHigher.trend.direction) || '--');
                setText('mtf-confluence', (mtf.confluence_pct || 0) + '%');
                setText('mtf-alignment', (mtf.alignment_score || 0).toFixed(2));
                setText('mtf-recommendation', mtf.recommendation || '--');
                
                // AI Filter
                var ai = data.ai || {};
                setText('ai-mode', ai.mode || 'filter');
                setText('ai-threshold', ((ai.threshold || 0.7) * 100).toFixed(0) + '%');
                setText('ai-approved', ai.approved || 0);
                setText('ai-rejected', ai.rejected || 0);
                var total = (ai.approved || 0) + (ai.rejected || 0);
                setText('ai-rate', total > 0 ? ((ai.approved || 0) / total * 100).toFixed(0) + '%' : '--');
                setText('ai-last', ai.last_decision || '--');
                
                // ML Model Status
                var ml = data.ml || {};
                var mlBadge = document.getElementById('ml-status-badge');
                if (mlBadge) {
                    if (ml.loaded) {
                        mlBadge.textContent = 'ACTIVE';
                        mlBadge.className = 'badge badge-success';
                        setText('ml-status', ml.status || 'Loaded');
                        setText('ml-accuracy', ((ml.accuracy || 0) * 100).toFixed(1) + '%');
                        setText('ml-samples', ml.samples || 0);
                        setText('ml-features', (ml.features || 24) + ' features');
                        if (ml.last_prediction) {
                            setText('ml-last-pred', ((ml.last_prediction.probability || 0.5) * 100).toFixed(0) + '% win');
                        }
                    } else {
                        mlBadge.textContent = 'NOT LOADED';
                        mlBadge.className = 'badge badge-neutral';
                        setText('ml-status', 'Not Available');
                        setText('ml-accuracy', '--');
                        setText('ml-samples', '--');
                        setText('ml-features', '--');
                        setText('ml-last-pred', '--');
                    }
                }
                
                // AI Decision Tracker
                var tracker = data.ai_tracker || {};
                setText('tracker-total', tracker.total_tracked || 0);
                setText('tracker-approval-rate', tracker.approval_rate || '--');
                var accEl = document.getElementById('tracker-approval-accuracy');
                if (accEl) {
                    var accVal = tracker.approval_accuracy || 'N/A';
                    accEl.textContent = accVal;
                    if (accVal !== 'N/A') {
                        var accNum = parseFloat(accVal);
                        if (accNum >= 60) accEl.className = 'indicator-value positive';
                        else if (accNum >= 50) accEl.className = 'indicator-value warning';
                        else accEl.className = 'indicator-value negative';
                    }
                }
                var netEl = document.getElementById('tracker-net-value');
                if (netEl) {
                    var netVal = tracker.net_ai_value || '$0.00';
                    netEl.textContent = netVal;
                    netEl.className = 'indicator-value ' + (netVal.includes('+') ? 'positive' : netVal.includes('-') ? 'negative' : '');
                }
                
                // Pre-Filter Statistics
                var pf = data.prefilter || {};
                setText('prefilter-total', pf.total_signals || 0);
                setText('prefilter-passed', pf.passed || 0);
                setText('prefilter-score', pf.blocked_by_score || 0);
                setText('prefilter-adx-low', pf.blocked_by_adx_low || 0);
                setText('prefilter-adx-danger', pf.blocked_by_adx_danger || 0);
                setText('prefilter-volume', pf.blocked_by_volume || 0);
                setText('prefilter-confluence', pf.blocked_by_confluence || 0);
                var pfBadge = document.getElementById('prefilter-badge');
                if (pfBadge) {
                    var passRate = pf.pass_rate || '0%';
                    pfBadge.textContent = passRate + ' PASS';
                    var passNum = parseFloat(passRate);
                    if (passNum >= 70) pfBadge.className = 'badge badge-success';
                    else if (passNum >= 40) pfBadge.className = 'badge badge-warning';
                    else pfBadge.className = 'badge badge-danger';
                }
                
                // Position
                var posDisplay = document.getElementById('position-display');
                var pos = data.position;
                if (posDisplay) {
                    if (pos && pos.symbol) {
                        var pnlClass = (pos.pnl || 0) >= 0 ? 'positive' : 'negative';
                        var boxClass = pos.side === 'LONG' ? 'long' : 'short';
                        posDisplay.innerHTML = '<div class="signal-box ' + boxClass + '">' +
                            '<div class="signal-label">' + pos.symbol + '</div>' +
                            '<div class="signal-value">' + pos.side + '</div>' +
                            '</div>' +
                            '<div style="margin-top: 10px;">' +
                            '<div class="indicator-row"><span class="indicator-name">Entry</span><span class="indicator-value">$' + pos.entry.toFixed(4) + '</span></div>' +
                            '<div class="indicator-row"><span class="indicator-name">Size</span><span class="indicator-value">' + pos.size.toFixed(4) + '</span></div>' +
                            '<div class="indicator-row"><span class="indicator-name">Unrealized P&L</span><span class="indicator-value ' + pnlClass + '">' + (pos.pnl >= 0 ? '+' : '') + '$' + pos.pnl.toFixed(2) + '</span></div>' +
                            '<div class="indicator-row"><span class="indicator-name">Stop Loss</span><span class="indicator-value negative">$' + (pos.stop_loss || 0).toFixed(4) + '</span></div>' +
                            '</div>';
                    } else {
                        posDisplay.innerHTML = '<div class="signal-box neutral"><div class="signal-value">NO POSITION</div></div>';
                    }
                }
            
                // Trading Parameters
                var params = data.params || {};
                setText('param-risk', ((params.risk_pct || 0.02) * 100).toFixed(1) + '%');
                setText('param-atr', (params.atr_mult || 1.5) + 'x');
                setText('param-tp', (params.tp1_r || 1.5) + 'R / ' + (params.tp2_r || 2.5) + 'R / ' + (params.tp3_r || 4) + 'R');
                setText('param-sizing', ((params.tp1_pct || 0.4) * 100) + '% / ' + ((params.tp2_pct || 0.35) * 100) + '% / ' + ((params.tp3_pct || 0.25) * 100) + '%');
                setText('param-trail', (params.trail_trigger_r || 1.5) + 'R');
                
                // Signals Table
                var signalsBody = document.getElementById('signals-body');
                var signals = data.signals || [];
                if (signalsBody) {
                    if (signals.length === 0) {
                        signalsBody.innerHTML = '<tr><td colspan="6" style="color: #666;">No signals yet</td></tr>';
                    } else {
                        var signalsHtml = '';
                        signals.slice(-10).reverse().forEach(function(sig) {
                            var dirClass = sig.direction === 'LONG' ? 'positive' : sig.direction === 'SHORT' ? 'negative' : '';
                            var statusBadge = sig.executed ? '<span class="badge badge-success">EXECUTED</span>' : 
                                               sig.rejected ? '<span class="badge badge-danger">REJECTED</span>' : 
                                               '<span class="badge badge-neutral">PENDING</span>';
                            signalsHtml += '<tr class="trade-row">' +
                                '<td>' + (sig.time || 'N/A') + '</td>' +
                                '<td class="' + dirClass + '">' + (sig.direction || 'N/A') + '</td>' +
                                '<td>' + ((sig.confidence || 0) * 100).toFixed(0) + '%</td>' +
                                '<td>$' + (sig.entry || 0).toFixed(4) + '</td>' +
                                '<td>' + (sig.ai_decision || '--') + '</td>' +
                                '<td>' + statusBadge + '</td>' +
                                '</tr>';
                        });
                        signalsBody.innerHTML = signalsHtml;
                    }
                }
                
                // Trades Table
                var tradesBody = document.getElementById('trades-body');
                var trades = data.trades || [];
                if (tradesBody) {
                    if (trades.length === 0) {
                        tradesBody.innerHTML = '<tr><td colspan="6" style="color: #666;">No trades yet</td></tr>';
                    } else {
                        var tradesHtml = '';
                        trades.slice(-10).reverse().forEach(function(trade) {
                            var pnlClass = (trade.pnl || 0) >= 0 ? 'positive' : 'negative';
                            var resultBadge = (trade.pnl || 0) >= 0 ? 
                                '<span class="badge badge-success">WIN</span>' : 
                                '<span class="badge badge-danger">LOSS</span>';
                            tradesHtml += '<tr class="trade-row">' +
                                '<td>' + (trade.time || 'N/A') + '</td>' +
                                '<td>' + (trade.side || 'N/A') + '</td>' +
                                '<td>$' + (trade.entry || 0).toFixed(4) + '</td>' +
                                '<td>$' + (trade.exit || 0).toFixed(4) + '</td>' +
                                '<td class="' + pnlClass + '">' + (trade.pnl >= 0 ? '+' : '') + '$' + Math.abs(trade.pnl || 0).toFixed(2) + '</td>' +
                                '<td>' + resultBadge + '</td>' +
                                '</tr>';
                        });
                        tradesBody.innerHTML = tradesHtml;
                    }
                }
                
                // Equity Chart
                updateEquityChart(data.equity_curve || []);
            } catch (err) {
                console.error('ERROR in updateDashboard:', err.message, err.stack);
            }
        }
        
        function updateMTF(id, direction) {
            var el = document.getElementById(id);
            if (el) {
                el.textContent = (direction || '--').toUpperCase();
                el.className = 'mtf-trend ' + (direction === 'bullish' ? 'bullish' : direction === 'bearish' ? 'bearish' : 'neutral');
            }
        }
        
        function updateEquityChart(equityCurve) {
            try {
                var canvas = document.getElementById('equityChart');
                if (!canvas) return;
                var ctx = canvas.getContext('2d');
                if (!equityCurve || equityCurve.length === 0) return;
                
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
            } catch (err) {
                console.error('Error in updateEquityChart:', err.message);
            }
        }
        
        // Initial data fetch
        fetchData();
        fetchPriceData();
        fetchLogs();
        fetchMarketScan();
        
        // Auto-refresh intervals
        setInterval(fetchData, 5000);        // Main data every 5 seconds
        setInterval(fetchPriceData, 3000);   // Price chart every 3 seconds
        setInterval(fetchLogs, 10000);       // Logs every 10 seconds
        setInterval(fetchMarketScan, 15000); // Market scan every 15 seconds
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
        # AI explanation callback
        self.get_ai_explanation = None
        # Market scanner callbacks
        self.get_market_scan = None
        self.switch_symbol = None
        self.ai_analyze_markets = None
        # NEW: AI tracker and pre-filter stats
        self.get_ai_tracker_stats = None
        self.get_prefilter_stats = None
        # Full state callback
        self.get_full_state = None
        
        if FLASK_AVAILABLE:
            self._setup_flask()
    
    def _setup_flask(self):
        """Setup Flask application."""
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)
        
        # Suppress Flask logs
        import logging as flask_logging
        flask_logging.getLogger('werkzeug').setLevel(flask_logging.WARNING)
        
        # Add CORS headers to all responses
        @self.app.after_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        @self.app.route('/debug')
        def debug_page():
            """Ultra-minimal debug page."""
            return '''<!DOCTYPE html>
<html>
<head><title>Debug</title></head>
<body style="background:#000;color:#0f0;font-family:monospace;padding:20px;font-size:14px;">
<h1>JULABA DEBUG PAGE</h1>
<div id="log" style="white-space:pre-wrap;"></div>
<script>
function log(msg) {
    document.getElementById('log').textContent += new Date().toISOString() + ' - ' + msg + '\\n';
    console.log(msg);
}
log('Script started');
log('Window location: ' + window.location.href);
log('Protocol: ' + window.location.protocol);

log('Attempting fetch to /api/data...');
var xhr = new XMLHttpRequest();
xhr.open('GET', '/api/data', true);
xhr.onreadystatechange = function() {
    log('XHR state: ' + xhr.readyState + ', status: ' + xhr.status);
    if (xhr.readyState === 4) {
        if (xhr.status === 200) {
            log('SUCCESS! Response length: ' + xhr.responseText.length);
            try {
                var data = JSON.parse(xhr.responseText);
                log('Parsed JSON successfully');
                log('Balance: $' + (data.balance ? data.balance.current : 'N/A'));
                log('Price: $' + (data.status ? data.status.current_price : 'N/A'));
                log('Symbol: ' + (data.status ? data.status.symbol : 'N/A'));
            } catch(e) {
                log('JSON parse error: ' + e.message);
            }
        } else {
            log('HTTP Error: ' + xhr.status + ' ' + xhr.statusText);
        }
    }
};
xhr.onerror = function() {
    log('XHR ERROR: Network error occurred');
};
xhr.send();
log('XHR request sent');
</script>
</body>
</html>'''
        
        @self.app.route('/test')
        def test_page():
            """Simple test page to verify API works."""
            return '''<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body style="background:#111;color:#fff;font-family:monospace;padding:20px;">
<h1>Dashboard Test</h1>
<div id="result">Loading...</div>
<script>
fetch('/api/data')
  .then(r => r.json())
  .then(d => {
    document.getElementById('result').innerHTML = 
      '<p>Status: ' + (d.status ? d.status.connected : 'N/A') + '</p>' +
      '<p>Price: $' + (d.status ? d.status.current_price : 'N/A') + '</p>' +
      '<p>Balance: $' + (d.balance ? d.balance.current : 'N/A') + '</p>' +
      '<p>Regime: ' + (d.regime ? d.regime.regime : 'N/A') + '</p>' +
      '<pre>' + JSON.stringify(d, null, 2).substring(0, 2000) + '</pre>';
  })
  .catch(e => {
    document.getElementById('result').textContent = 'Error: ' + e.message;
  });
</script>
</body>
</html>'''

        @self.app.route('/simple')
        def simple_dashboard():
            """Simplified working dashboard."""
            return '''<!DOCTYPE html>
<html>
<head>
    <title>Julaba - Simple</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 100%); color: #fff; min-height: 100vh; padding: 20px; }
        h1 { text-align: center; color: #00d4ff; font-size: 2.5em; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; max-width: 1200px; margin: 0 auto; }
        .card { background: rgba(0,30,60,0.8); border: 1px solid #00d4ff44; border-radius: 12px; padding: 20px; }
        .card h2 { color: #00d4ff; font-size: 1.2em; margin-bottom: 15px; border-bottom: 1px solid #00d4ff44; padding-bottom: 10px; }
        .row { display: flex; justify-content: space-between; margin: 8px 0; }
        .label { color: #888; }
        .value { color: #fff; font-weight: bold; }
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .status { text-align: center; padding: 10px; background: #00d4ff22; border-radius: 8px; margin-bottom: 20px; }
        .refresh-btn { display: block; width: 200px; margin: 30px auto; padding: 15px; background: linear-gradient(135deg, #00d4ff, #0088cc); color: #fff; border: none; border-radius: 8px; font-size: 1.1em; cursor: pointer; }
        .refresh-btn:hover { background: linear-gradient(135deg, #00e5ff, #00aaff); }
    </style>
</head>
<body>
    <h1>🚀 JULABA Trading Dashboard</h1>
    <div class="status" id="status-bar">Loading...</div>
    <div class="grid">
        <div class="card">
            <h2>💰 Balance</h2>
            <div class="row"><span class="label">Current:</span><span class="value" id="balance">-</span></div>
            <div class="row"><span class="label">Initial:</span><span class="value" id="initial">-</span></div>
            <div class="row"><span class="label">P&L:</span><span class="value" id="pnl">-</span></div>
        </div>
        <div class="card">
            <h2>📊 Market</h2>
            <div class="row"><span class="label">Price:</span><span class="value" id="price">-</span></div>
            <div class="row"><span class="label">Regime:</span><span class="value" id="regime">-</span></div>
            <div class="row"><span class="label">Hurst:</span><span class="value" id="hurst">-</span></div>
        </div>
        <div class="card">
            <h2>🤖 AI Status</h2>
            <div class="row"><span class="label">Mode:</span><span class="value" id="ai-mode">-</span></div>
            <div class="row"><span class="label">Signals:</span><span class="value" id="ai-signals">-</span></div>
            <div class="row"><span class="label">Approved:</span><span class="value" id="ai-approved">-</span></div>
        </div>
        <div class="card">
            <h2>📈 Position</h2>
            <div class="row"><span class="label">Side:</span><span class="value" id="pos-side">-</span></div>
            <div class="row"><span class="label">Size:</span><span class="value" id="pos-size">-</span></div>
            <div class="row"><span class="label">Entry:</span><span class="value" id="pos-entry">-</span></div>
        </div>
        <div class="card">
            <h2>🎯 Trading</h2>
            <div class="row"><span class="label">Total Trades:</span><span class="value" id="total-trades">-</span></div>
            <div class="row"><span class="label">Win Rate:</span><span class="value" id="win-rate">-</span></div>
            <div class="row"><span class="label">Symbol:</span><span class="value" id="symbol">-</span></div>
        </div>
        <div class="card">
            <h2>⚙️ Parameters</h2>
            <div class="row"><span class="label">Timeframe:</span><span class="value" id="timeframe">-</span></div>
            <div class="row"><span class="label">Risk/Trade:</span><span class="value" id="risk">-</span></div>
            <div class="row"><span class="label">Stop Loss:</span><span class="value" id="stop-loss">-</span></div>
        </div>
    </div>
    <button class="refresh-btn" onclick="loadData()">🔄 Refresh Data</button>
    
    <script>
        function loadData() {
            document.getElementById('status-bar').textContent = 'Refreshing...';
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/data', true);
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    try {
                        var d = JSON.parse(xhr.responseText);
                        // Status
                        if (d.status) {
                            document.getElementById('status-bar').textContent = (d.status.connected ? '🟢 Connected' : '🔴 Disconnected') + ' | ' + (d.status.symbol || 'N/A') + ' | Last update: ' + new Date().toLocaleTimeString();
                            document.getElementById('price').textContent = '$' + (d.status.current_price || 0).toFixed(4);
                            document.getElementById('symbol').textContent = d.status.symbol || 'N/A';
                        }
                        // Balance
                        if (d.balance) {
                            document.getElementById('balance').textContent = '$' + (d.balance.current || 0).toFixed(2);
                            document.getElementById('initial').textContent = '$' + (d.balance.initial || 0).toFixed(2);
                        }
                        // PnL
                        if (d.pnl) {
                            var pnlEl = document.getElementById('pnl');
                            var pnlVal = d.pnl.total || 0;
                            pnlEl.textContent = (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toFixed(2) + ' (' + (d.pnl.total_pct || 0).toFixed(2) + '%)';
                            pnlEl.className = 'value ' + (pnlVal >= 0 ? 'positive' : 'negative');
                        }
                        // Regime
                        if (d.regime) {
                            document.getElementById('regime').textContent = d.regime.regime || 'N/A';
                            document.getElementById('hurst').textContent = (d.regime.hurst || 0).toFixed(3);
                        }
                        // AI
                        if (d.ai) {
                            document.getElementById('ai-mode').textContent = d.ai.mode || 'N/A';
                            document.getElementById('ai-signals').textContent = d.ai.signals_analyzed || 0;
                            document.getElementById('ai-approved').textContent = d.ai.approved || 0;
                        }
                        // Position
                        if (d.position) {
                            document.getElementById('pos-side').textContent = d.position.side || 'NONE';
                            document.getElementById('pos-size').textContent = (d.position.size || 0).toFixed(4);
                            document.getElementById('pos-entry').textContent = d.position.entry_price ? '$' + d.position.entry_price.toFixed(4) : '-';
                        } else {
                            document.getElementById('pos-side').textContent = 'NONE';
                            document.getElementById('pos-size').textContent = '0';
                            document.getElementById('pos-entry').textContent = '-';
                        }
                        // Trades
                        if (d.trades) {
                            document.getElementById('total-trades').textContent = d.trades.total || 0;
                            document.getElementById('win-rate').textContent = (d.trades.win_rate || 0).toFixed(1) + '%';
                        }
                        // Params
                        if (d.params) {
                            document.getElementById('timeframe').textContent = d.params.timeframe || 'N/A';
                            document.getElementById('risk').textContent = ((d.params.risk_pct || 0) * 100) + '%';
                            document.getElementById('stop-loss').textContent = (d.params.atr_mult || 0) + 'x ATR';
                        }
                    } catch(e) {
                        document.getElementById('status-bar').textContent = '🔴 Parse Error: ' + e.message;
                    }
                } else if (xhr.readyState === 4) {
                    document.getElementById('status-bar').textContent = '🔴 Error: ' + xhr.status;
                }
            };
            xhr.onerror = function() {
                document.getElementById('status-bar').textContent = '🔴 Network Error';
            };
            xhr.send();
        }
        // Load immediately and every 5 seconds
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>'''
        
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
                # NEW: AI tracker and pre-filter stats
                if self.get_ai_tracker_stats:
                    data['ai_tracker'] = self.get_ai_tracker_stats()
                if self.get_prefilter_stats:
                    data['prefilter'] = self.get_prefilter_stats()
            except Exception as e:
                logger.error(f"Dashboard API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/state')
        def api_state():
            """Unified system state - SINGLE SOURCE OF TRUTH for all components."""
            data = {}
            try:
                if self.get_full_state:
                    data = self.get_full_state()
                else:
                    data['error'] = 'Full state not available'
            except Exception as e:
                logger.error(f"Dashboard state API error: {e}")
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
        
        @self.app.route('/api/ai-explain', methods=['POST'])
        def api_ai_explain():
            """Get AI explanation for a dashboard topic."""
            data = {}
            
            try:
                req_data = request.get_json() or {}
                topic = req_data.get('topic', '')
                display_name = req_data.get('display_name', topic)
                
                if not topic:
                    return jsonify({'error': 'No topic specified'})
                
                if self.get_ai_explanation:
                    explanation = self.get_ai_explanation(topic, display_name)
                    data['explanation'] = explanation
                    data['topic'] = topic
                else:
                    data['error'] = 'AI explanation service not available'
            except Exception as e:
                logger.error(f"Dashboard AI explain API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/market-scan')
        def api_market_scan():
            """Get multi-pair market data for scanner."""
            data = {'pairs': [], 'current_symbol': '', 'multi_pair_enabled': False, 'multi_pair_count': 0}
            
            try:
                if self.get_market_scan:
                    scan_data = self.get_market_scan()
                    data['pairs'] = scan_data.get('pairs', [])
                    data['current_symbol'] = scan_data.get('current_symbol', '')
                    data['multi_pair_enabled'] = scan_data.get('multi_pair_enabled', False)
                    data['multi_pair_count'] = scan_data.get('multi_pair_count', 0)
                    data['active_pairs'] = scan_data.get('active_pairs', [])
            except Exception as e:
                logger.error(f"Dashboard market scan API error: {e}")
                data['error'] = str(e)
            
            return jsonify(data)
        
        @self.app.route('/api/switch-symbol', methods=['POST'])
        def api_switch_symbol():
            """Switch trading symbol."""
            data = {}
            
            try:
                req_data = request.get_json() or {}
                symbol = req_data.get('symbol', '')
                
                if not symbol:
                    return jsonify({'error': 'No symbol specified', 'success': False})
                
                if self.switch_symbol:
                    result = self.switch_symbol(symbol)
                    data['success'] = result.get('success', False)
                    data['message'] = result.get('message', '')
                    if not result.get('success'):
                        data['error'] = result.get('error', 'Unknown error')
                else:
                    data['error'] = 'Symbol switch not available'
                    data['success'] = False
            except Exception as e:
                logger.error(f"Dashboard switch symbol API error: {e}")
                data['error'] = str(e)
                data['success'] = False
            
            return jsonify(data)
        
        @self.app.route('/api/ai-analyze-markets', methods=['POST'])
        def api_ai_analyze_markets():
            """Get AI analysis of all market pairs."""
            data = {}
            
            try:
                if self.ai_analyze_markets:
                    result = self.ai_analyze_markets()
                    data['recommendation'] = result.get('recommendation', '')
                    data['best_pair'] = result.get('best_pair', '')
                else:
                    data['error'] = 'AI market analysis not available'
            except Exception as e:
                logger.error(f"Dashboard AI analyze markets API error: {e}")
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
