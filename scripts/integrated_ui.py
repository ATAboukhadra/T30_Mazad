"""Integrated UI for all 3 stages: Video playback + ASR + Player Matching + LLM Check.

Stage 1: Video selection (upload/file picker), playback, live ASR transcription
Stage 2: Token-to-player matching with suggestions
Stage 3: LLM validation of selected players against the question

Usage:
    python scripts/integrated_ui.py --player-db data/players_enriched.jsonl
    # Then upload video through the web interface
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template_string, request, send_file, send_from_directory

from stage2_match_names import load_players, process_pass, fuzzy_match, normalize
from verify_names import load_player_database, verify_with_llm

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global state
STATE: Dict[str, Any] = {
    "video_path": None,
    "player_db_path": None,
    "question": "",
    "start_time": 0.0,
    "end_time": 0.0,
    "whisper_model": "large",
    "asr_running": False,
    "asr_result": None,
    "tokens": [],
    "suggestions": {},
    "selections": {},
    "llm_results": {},
    "llm_client": None,
    "llm_provider": "gemini",
    "llm_model": None,
    "player_db": None,
    "players_by_name": None,
    "all_names": None,
    "uploaded_videos": {},  # Store uploaded video paths
}

STAGE2_MIN_GRAM = 1
STAGE2_MAX_GRAM = 3
STAGE2_FUZZY_THRESHOLD = 70
STAGE2_MAX_SUGGESTIONS = 5

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Football Quiz Verifier</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        header h1 {
            font-size: 32px;
            color: #fff;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #ddd;
            font-size: 14px;
        }
        
        /* Video upload section */
        .video-upload-section {
            background: #0f3460;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .video-upload-section.has-video {
            display: none;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 50px 20px;
            cursor: pointer;
            transition: all 0.3s;
            background: #1a2332;
        }
        .upload-area:hover {
            border-color: #00d4ff;
            background: #1f2c40;
        }
        .upload-area.dragging {
            border-color: #00ff88;
            background: #1a3a2e;
        }
        .upload-icon {
            font-size: 64px;
            margin-bottom: 20px;
        }
        .upload-text {
            font-size: 18px;
            color: #ddd;
            margin-bottom: 10px;
        }
        .upload-subtext {
            font-size: 13px;
            color: #888;
        }
        #file-input {
            display: none;
        }
        .btn-browse {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-browse:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .upload-progress {
            display: none;
            margin-top: 20px;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #1a2332;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #00d4ff);
            width: 0%;
            transition: width 0.3s;
        }
        .progress-text {
            color: #00d4ff;
            font-size: 13px;
        }
        
        /* Stage tabs */
        .stage-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            background: #16213e;
            padding: 10px;
            border-radius: 10px;
        }
        .stage-tab {
            flex: 1;
            padding: 15px;
            background: #0f3460;
            border: none;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            position: relative;
        }
        .stage-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .stage-tab.completed::after {
            content: '✓';
            position: absolute;
            top: 5px;
            right: 10px;
            color: #00ff88;
            font-size: 18px;
        }
        .stage-tab:hover:not(.active) {
            background: #1a4a7e;
            color: #fff;
        }
        
        /* Stage content */
        .stage-content {
            display: none;
            background: #16213e;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .stage-content.active {
            display: block;
        }
        
        /* Stage 1: Video & ASR */
        .stage1-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 1200px) {
            .stage1-container {
                grid-template-columns: 1fr;
            }
        }
        .video-section, .asr-section {
            background: #0f3460;
            padding: 20px;
            border-radius: 10px;
        }
        .section-title {
            font-size: 18px;
            color: #00d4ff;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .video-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #1a2332;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 13px;
        }
        .video-name {
            color: #00d4ff;
            font-weight: 600;
        }
        .btn-change-video {
            padding: 5px 12px;
            background: #667eea;
            border: none;
            border-radius: 4px;
            color: #fff;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }
        .btn-change-video:hover {
            background: #7c8ff0;
        }
        video {
            width: 100%;
            border-radius: 8px;
            background: #000;
            max-height: 400px;
        }
        .video-controls {
            margin-top: 15px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .control-group label {
            font-size: 12px;
            color: #888;
        }
        .control-group input {
            padding: 10px;
            background: #16213e;
            border: 1px solid #2a4a7e;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
        }
        .control-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .playback-controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .playback-controls button {
            flex: 1;
            padding: 10px;
            background: #667eea;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .playback-controls button:hover {
            background: #7c8ff0;
            transform: translateY(-2px);
        }
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }
        .speed-control input[type="range"] {
            flex: 1;
        }
        .speed-value {
            min-width: 60px;
            text-align: center;
            color: #00d4ff;
            font-weight: 600;
        }
        
        /* Question input */
        .question-section {
            margin-top: 20px;
        }
        .question-section textarea {
            width: 100%;
            min-height: 80px;
            padding: 12px;
            background: #16213e;
            border: 1px solid #2a4a7e;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
            resize: vertical;
        }
        .question-section textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        /* ASR section */
        .btn-run-asr {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
            border: none;
            border-radius: 8px;
            color: #000;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.3s;
        }
        .btn-run-asr:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 255, 136, 0.3);
        }
        .btn-run-asr:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }
        .asr-status {
            margin-top: 15px;
            padding: 12px;
            background: #1a2332;
            border-radius: 6px;
            font-size: 13px;
            color: #00d4ff;
            text-align: center;
        }
        .asr-status.running {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .transcript-box {
            margin-top: 15px;
            padding: 15px;
            background: #1a2332;
            border: 1px solid #2a2f45;
            border-radius: 8px;
            width: 100%;
            min-height: 150px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: #ddd;
            resize: vertical;
            outline: none;
        }
        .transcript-box::-webkit-scrollbar {
            width: 6px;
        }
        .transcript-box::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 3px;
        }
        .live-word {
            display: inline-block;
            margin: 2px;
            padding: 2px 4px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 3px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Stage 2: Token matching */
        .tokens-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 15px;
        }
        .token-card {
            background: #0f3460;
            border-radius: 10px;
            padding: 15px;
            border: 2px solid transparent;
            transition: all 0.2s;
        }
        .token-card:hover {
            border-color: #667eea;
        }
        .token-card.matched {
            border-color: #00ff88;
            background: #1a3a2e;
        }
        .token-card.no-match {
            border-color: #ff4444;
            background: #2e1a1a;
        }
        .token-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .token-text {
            font-size: 22px;
            font-weight: bold;
            color: #fff;
        }
        .token-index {
            font-size: 12px;
            color: #888;
            background: #1a2332;
            padding: 3px 8px;
            border-radius: 4px;
        }
        .suggestions-list {
            max-height: 220px;
            overflow-y: auto;
            margin: 10px 0;
        }
        .suggestions-list::-webkit-scrollbar {
            width: 6px;
        }
        .suggestions-list::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 3px;
        }
        .search-box {
            margin-top: 10px;
            padding: 8px;
            background: #10162a;
            border: 1px solid #1f2742;
            border-radius: 6px;
        }
        .search-input {
            width: 100%;
            padding: 8px 10px;
            border-radius: 6px;
            border: 1px solid #2a2f45;
            background: #0d1326;
            color: #ddd;
            font-size: 12px;
        }
        .search-results {
            margin-top: 8px;
            max-height: 140px;
            overflow-y: auto;
        }
        .search-result {
            padding: 6px 8px;
            border-radius: 4px;
            background: #141b33;
            margin-bottom: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .search-result:hover {
            background: #1b2544;
        }
        .search-result-name {
            font-weight: bold;
            font-size: 12px;
            color: #fff;
        }
        .search-result-meta {
            font-size: 11px;
            color: #9aa3c7;
            margin-top: 2px;
        }
        .suggestion {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin: 5px 0;
            background: #1a2332;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .suggestion:hover {
            background: #2a4a7e;
            transform: translateX(3px);
        }
        .suggestion.selected {
            background: #667eea;
            color: #fff;
        }
        .suggestion-info {
            flex: 1;
        }
        .suggestion-name {
            font-weight: 600;
            font-size: 14px;
        }
        .suggestion-meta {
            font-size: 11px;
            color: #888;
            margin-top: 2px;
        }
        .suggestion.selected .suggestion-meta {
            color: #ddd;
        }
        .token-actions {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .btn {
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.15s;
            font-weight: 600;
        }
        .btn-no-match {
            background: #ff4444;
            color: #fff;
            flex: 1;
        }
        .btn-no-match:hover {
            background: #ff6666;
        }
        .btn-clear {
            background: #555;
            color: #fff;
        }
        .btn-clear:hover {
            background: #777;
        }
        
        /* Stage 3: LLM Check */
        .llm-controls {
            margin-bottom: 20px;
            padding: 20px;
            background: #0f3460;
            border-radius: 10px;
        }
        .btn-run-llm {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-run-llm:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(245, 87, 108, 0.3);
        }
        .btn-run-llm:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }
        .llm-results {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 15px;
        }
        .llm-card {
            background: #0f3460;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid transparent;
        }
        .llm-card.yes {
            border-left-color: #00ff88;
        }
        .llm-card.no {
            border-left-color: #ff4444;
        }
        .llm-card.uncertain {
            border-left-color: #ffaa00;
        }
        .llm-player-name {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .llm-verdict {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .llm-verdict-icon {
            font-size: 28px;
        }
        .llm-verdict-text {
            font-size: 16px;
            font-weight: 600;
        }
        .llm-justification {
            padding: 12px;
            background: #1a2332;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.5;
            color: #ccc;
        }
        
        /* Stats bar */
        .stats-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .stat-box {
            flex: 1;
            min-width: 150px;
            padding: 15px;
            background: #0f3460;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #00d4ff;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        
        /* Action buttons */
        .action-buttons {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            justify-content: center;
        }
        .btn-action {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .btn-secondary {
            background: #2a4a7e;
            color: #fff;
        }
        .btn-secondary:hover {
            background: #3a5a9e;
        }
        
        /* Loading spinner */
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* No suggestions message */
        .no-suggestions {
            text-align: center;
            padding: 20px;
            color: #888;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚽ Football Quiz Verifier</h1>
            <p class="subtitle">Integrated Video Analysis, Player Matching & LLM Verification</p>
        </header>
        
        <!-- Video Upload Section -->
        <div class="video-upload-section" id="video-upload-section">
            <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
                <div class="upload-icon">📹</div>
                <div class="upload-text">Drop video file here or click to browse</div>
                <div class="upload-subtext">Supports MP4, MOV, AVI, MKV (max 500MB)</div>
                <button class="btn-browse" onclick="event.stopPropagation(); document.getElementById('file-input').click()">
                    Browse Files
                </button>
            </div>
            <input type="file" id="file-input" accept="video/*" onchange="handleFileSelect(event)">
            <div class="upload-progress" id="upload-progress">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="progress-text" id="progress-text">Uploading...</div>
            </div>
        </div>
        
        <!-- Stage Tabs -->
        <div class="stage-tabs" id="stage-tabs" style="display:none;">
            <button class="stage-tab active" data-stage="1" onclick="switchStage(1)">
                📹 Stage 1: Video & ASR
            </button>
            <button class="stage-tab" data-stage="2" onclick="switchStage(2)">
                🎯 Stage 2: Player Matching
            </button>
            <button class="stage-tab" data-stage="3" onclick="switchStage(3)">
                🤖 Stage 3: LLM Verification
            </button>
        </div>
        
        <!-- Stage 1: Video & ASR -->
        <div class="stage-content active" id="stage1" style="display:none;">
            <div class="stage1-container">
                <!-- Video Section -->
                <div class="video-section">
                    <div class="section-title">📹 Video Playback</div>
                    <div class="video-info">
                        <span class="video-name" id="video-filename">No video loaded</span>
                        <button class="btn-change-video" onclick="changeVideo()">Change Video</button>
                    </div>
                    <video id="video-player" controls>
                        <source id="video-source" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    
                    <div class="video-controls">
                        <div class="control-group">
                            <label>Start Time (mm:ss)</label>
                            <input type="text" id="start-time" placeholder="0:00" value="0:00">
                        </div>
                        <div class="control-group">
                            <label>End Time (mm:ss)</label>
                            <input type="text" id="end-time" placeholder="1:00" value="1:00">
                        </div>
                    </div>
                    
                    <div class="playback-controls">
                        <button onclick="setVideoTime()">⏱️ Set Times</button>
                        <button onclick="playSegment()">▶️ Play Segment</button>
                    </div>
                    
                    <div class="speed-control">
                        <label style="color: #888; font-size: 12px;">Playback Speed:</label>
                        <input type="range" id="speed-slider" min="25" max="200" value="100" 
                               oninput="updateSpeed(this.value)">
                        <span class="speed-value" id="speed-value">1.0x</span>
                    </div>
                    
                    <div class="question-section">
                        <label class="section-title">❓ Quiz Question</label>
                        <textarea id="question-input" placeholder="Enter the quiz question, e.g., 'Name 10 players who played for Barcelona'"></textarea>
                    </div>
                </div>
                
                <!-- ASR Section -->
                <div class="asr-section">
                    <div class="section-title">🎤 Speech Recognition</div>
                    <button class="btn-run-asr" id="btn-run-asr" onclick="runASR()">
                        🚀 Run ASR Transcription
                    </button>
                    <div class="asr-status" id="asr-status" style="display:none;"></div>
                    <textarea class="transcript-box" id="transcript-box" placeholder="Click &quot;Run ASR Transcription&quot; to start..."></textarea>
                    <div class="action-buttons" style="margin-top: 12px;">
                        <button class="btn-action btn-secondary" onclick="applyTranscriptEdits()">
                            ✍️ Apply Transcript Edits
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Stage 2: Token Matching -->
        <div class="stage-content" id="stage2">
            <div class="stats-bar">
                <div class="stat-box">
                    <div class="stat-value" id="total-tokens">0</div>
                    <div class="stat-label">Total Tokens</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="matched-tokens">0</div>
                    <div class="stat-label">Matched</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="no-match-tokens">0</div>
                    <div class="stat-label">No Match</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="pending-tokens">0</div>
                    <div class="stat-label">Pending</div>
                </div>
            </div>
            
            <div class="tokens-grid" id="tokens-grid">
                <div style="text-align: center; color: #666; padding: 50px;">
                    Run Stage 1 ASR first to see tokens...
                </div>
            </div>
            
            <div class="action-buttons">
                <button class="btn-action btn-secondary" onclick="switchStage(1)">
                    ← Back to Stage 1
                </button>
                <button class="btn-action btn-primary" onclick="proceedToStage3()">
                    Proceed to LLM Check →
                </button>
            </div>
        </div>
        
        <!-- Stage 3: LLM Verification -->
        <div class="stage-content" id="stage3">
            <div class="llm-controls">
                <div class="section-title">🤖 LLM Verification</div>
                <p style="color: #888; margin-bottom: 15px; font-size: 13px;">
                    The LLM will verify each selected player against your quiz question
                </p>
                <button class="btn-run-llm" id="btn-run-llm" onclick="runLLMCheck()">
                    🚀 Run LLM Verification
                </button>
            </div>
            
            <div class="llm-results" id="llm-results">
                <div style="text-align: center; color: #666; padding: 50px;">
                    Click "Run LLM Verification" to check players...
                </div>
            </div>
            <div class="asr-status" id="llm-summary" style="display:none;"></div>
            
            <div class="action-buttons">
                <button class="btn-action btn-secondary" onclick="switchStage(2)">
                    ← Back to Stage 2
                </button>
                <button class="btn-action btn-primary" onclick="exportResults()">
                    📥 Export Results
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let currentStage = 1;
        let videoPath = '';
        let videoFilename = '';
        let tokensData = [];
        let selections = {};
        let llmResults = {};
        let searchResults = {};
        let searchQueries = {};
        let searchTimers = {};
        
        // Initialize
        async function init() {
            const response = await fetch('/api/init');
            const data = await response.json();
            if (data.question) {
                document.getElementById('question-input').value = data.question;
            }
            if (data.transcript) {
                document.getElementById('transcript-box').value = data.transcript;
            }
        }
        
        // Drag and drop handlers
        const uploadArea = document.getElementById('upload-area');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragging');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragging');
            }, false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }, false);
        
        // File selection handler
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        // Upload video file
        async function handleFile(file) {
            if (!file.type.startsWith('video/')) {
                const progressText = document.getElementById('progress-text');
                progressText.textContent = '❌ Please select a video file';
                return;
            }
            
            const progress = document.getElementById('upload-progress');
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            
            progress.style.display = 'block';
            progressText.textContent = 'Uploading...';
            
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                const xhr = new XMLHttpRequest();
                
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        progressFill.style.width = percentComplete + '%';
                        progressText.textContent = `Uploading... ${Math.round(percentComplete)}%`;
                    }
                });
                
                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        const data = JSON.parse(xhr.responseText);
                        if (data.status === 'ok') {
                            progressText.textContent = '✅ Upload complete!';
                            videoPath = data.video_path;
                            videoFilename = data.filename;
                            
                            setTimeout(() => {
                                loadVideo();
                            }, 500);
                        } else {
                            progressText.textContent = '❌ Upload failed: ' + (data.error || 'Unknown error');
                        }
                    } else {
                        progressText.textContent = '❌ Upload failed';
                    }
                });
                
                xhr.addEventListener('error', () => {
                    progressText.textContent = '❌ Upload failed';
                });
                
                xhr.open('POST', '/api/upload-video');
                xhr.send(formData);
                
            } catch (error) {
                progressText.textContent = '❌ Upload failed: ' + error.message;
            }
        }
        
        // Load video after upload
        function loadVideo() {
            document.getElementById('video-upload-section').classList.add('has-video');
            document.getElementById('stage-tabs').style.display = 'flex';
            document.getElementById('stage1').style.display = 'block';
            
            document.getElementById('video-filename').textContent = videoFilename;
            document.getElementById('video-source').src = '/api/video';
            document.getElementById('video-player').load();
        }
        
        // Change video
        function changeVideo() {
            document.getElementById('video-upload-section').classList.remove('has-video');
            document.getElementById('stage-tabs').style.display = 'none';
            document.getElementById('stage1').style.display = 'none';
            document.getElementById('file-input').value = '';
            document.getElementById('upload-progress').style.display = 'none';
            document.getElementById('progress-fill').style.width = '0%';
            
            // Reset state
            tokensData = [];
            selections = {};
            llmResults = {};
            searchResults = {};
            searchQueries = {};
            document.getElementById('transcript-box').value = '';
            
            // Clear completed markers
            document.querySelectorAll('.stage-tab').forEach(tab => tab.classList.remove('completed'));
        }
        
        // Stage switching
        function switchStage(stage) {
            currentStage = stage;
            document.querySelectorAll('.stage-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.stage-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelector(`[data-stage="${stage}"]`).classList.add('active');
            document.getElementById(`stage${stage}`).classList.add('active');
        }
        
        // Video controls
        function parseTimestamp(ts) {
            const parts = ts.split(':');
            if (parts.length === 1) return parseFloat(parts[0]) || 0;
            if (parts.length === 2) return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
            return 0;
        }
        
        function setVideoTime() {
            const video = document.getElementById('video-player');
            const start = document.getElementById('start-time').value;
            video.currentTime = parseTimestamp(start);
        }
        
        function playSegment() {
            const video = document.getElementById('video-player');
            const start = parseTimestamp(document.getElementById('start-time').value);
            const end = parseTimestamp(document.getElementById('end-time').value);
            
            video.currentTime = start;
            video.play();
            
            const checkTime = setInterval(() => {
                if (video.currentTime >= end) {
                    video.pause();
                    clearInterval(checkTime);
                }
            }, 100);
        }
        
        function updateSpeed(value) {
            const speed = value / 100;
            document.getElementById('speed-value').textContent = speed.toFixed(1) + 'x';
            document.getElementById('video-player').playbackRate = speed;
        }
        
        // ASR
        async function runASR() {
            const question = document.getElementById('question-input').value;
            const startTime = document.getElementById('start-time').value;
            const endTime = document.getElementById('end-time').value;
            
            if (!question.trim()) {
                const status = document.getElementById('asr-status');
                status.style.display = 'block';
                status.className = 'asr-status';
                status.textContent = '❌ Please enter a quiz question first.';
                return;
            }
            
            const btn = document.getElementById('btn-run-asr');
            const status = document.getElementById('asr-status');
            const transcript = document.getElementById('transcript-box');
            
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running ASR...';
            status.style.display = 'block';
            status.className = 'asr-status running';
            status.textContent = '🎤 Transcribing audio... This may take a minute...';
            transcript.value = 'Processing...';
            
            try {
                const response = await fetch('/api/run-asr', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        start_time: startTime,
                        end_time: endTime
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'ok') {
                    status.className = 'asr-status';
                    status.textContent = '✅ Transcription complete!';
                    
                    transcript.value = data.transcript || '';
                    
                    // Mark stage 1 as complete
                    document.querySelector('[data-stage="1"]').classList.add('completed');
                    
                    // Load tokens for stage 2
                    loadTokens();
                    
                    setTimeout(() => {
                        switchStage(2);
                    }, 1500);
                } else {
                    status.className = 'asr-status';
                    status.textContent = '❌ Error: ' + (data.error || 'Unknown error');
                    transcript.value = 'Error running ASR';
                }
            } catch (error) {
                status.className = 'asr-status';
                status.textContent = '❌ Error: ' + error.message;
                transcript.value = 'Network error';
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🚀 Run ASR Transcription';
            }
        }
        
        // Token matching
        async function loadTokens() {
            const response = await fetch('/api/tokens');
            const data = await response.json();
            tokensData = data.tokens || [];
            selections = data.selections || {};
            renderTokens();
        }

        async function applyTranscriptEdits() {
            const transcript = document.getElementById('transcript-box').value.trim();
            if (!transcript) {
                const status = document.getElementById('asr-status');
                status.style.display = 'block';
                status.className = 'asr-status';
                status.textContent = '❌ Please enter a transcript first.';
                return;
            }
            const response = await fetch('/api/set-transcript', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ transcript })
            });
            const data = await response.json();
            if (data.status !== 'ok') {
                const status = document.getElementById('asr-status');
                status.style.display = 'block';
                status.className = 'asr-status';
                status.textContent = '❌ Failed to apply transcript: ' + (data.error || 'Unknown error');
                return;
            }
            await loadTokens();
            document.querySelector('[data-stage="1"]').classList.add('completed');
            switchStage(2);
        }
        
        function renderTokens() {
            const grid = document.getElementById('tokens-grid');
            if (tokensData.length === 0) {
                grid.innerHTML = '<div style="text-align: center; color: #666; padding: 50px;">No tokens found. Run Stage 1 first.</div>';
                return;
            }
            
            grid.innerHTML = '';
            tokensData.forEach((token, idx) => {
                const card = document.createElement('div');
                const selection = selections[idx];
                const status = selection === undefined ? 'pending' : (selection === null ? 'no-match' : 'matched');
                card.className = `token-card ${status}`;
                
                const suggestions = token.suggestions || [];
                
                card.innerHTML = `
                    <div class="token-header">
                        <span class="token-text">${escapeHtml(token.token)}</span>
                        <span class="token-index">#${idx + 1}</span>
                    </div>
                    <div class="suggestions-list">
                        ${suggestions.length > 0 ? suggestions.slice(0, 10).map(s => `
                            <div class="suggestion ${selection === s.name ? 'selected' : ''}" 
                                 onclick="selectPlayer(${idx}, '${escapeJs(s.name)}')">
                                <div class="suggestion-info">
                                    <div class="suggestion-name">${escapeHtml(s.name)}</div>
                                    <div class="suggestion-meta">
                                        ${s.player?.position || ''}
                                        ${s.player?.current_club ? '| ' + s.player.current_club : ''}
                                        ${s.player?.nationality ? '| ' + s.player.nationality : ''}
                                        ${s.match_type ? '| ' + s.match_type : ''}
                                        ${typeof s.score !== 'undefined' ? '| ' + s.score + '% match' : ''}
                                        ${typeof s.career_score !== 'undefined' ? '| ' + Math.round(s.career_score) + ' career' : ''}
                                    </div>
                                </div>
                            </div>
                        `).join('') : '<div class="no-suggestions">No suggestions found</div>'}
                    </div>
                    <div class="search-box">
                        <input class="search-input" type="text" placeholder="Search player..."
                               value="${escapeHtml(searchQueries[idx] || '')}"
                               oninput="searchPlayers(${idx}, this.value)">
                        <div class="search-results" id="search-results-${idx}">
                            ${renderSearchResults(idx)}
                        </div>
                    </div>
                    <div class="token-actions">
                        <button class="btn btn-no-match" onclick="selectPlayer(${idx}, null)">✗ No Match</button>
                        ${selection !== undefined ? `<button class="btn btn-clear" onclick="clearSelection(${idx})">Clear</button>` : ''}
                    </div>
                `;
                grid.appendChild(card);
            });
            
            updateStats();
        }

        function renderSearchResults(idx) {
            const results = searchResults[idx] || [];
            const query = (searchQueries[idx] || '').trim();
            if (!query) return '';
            if (results.length === 0) {
                return '<div class="no-suggestions">No matches</div>';
            }
            return results.slice(0, 8).map(r => `
                <div class="search-result" onclick="selectPlayer(${idx}, '${escapeJs(r.name)}')">
                    <div class="search-result-name">${escapeHtml(r.name)}</div>
                    <div class="search-result-meta">
                        ${r.player?.position || ''}
                        ${r.player?.current_club ? '| ' + r.player.current_club : ''}
                        ${r.player?.nationality ? '| ' + r.player.nationality : ''}
                        ${typeof r.score !== 'undefined' ? '| ' + r.score + '% match' : ''}
                        ${typeof r.career_score !== 'undefined' ? '| ' + Math.round(r.career_score) + ' career' : ''}
                    </div>
                </div>
            `).join('');
        }

        function searchPlayers(idx, query) {
            searchQueries[idx] = query;
            const target = document.getElementById(`search-results-${idx}`);
            if (!target) return;
            const trimmed = query.trim();
            if (trimmed.length < 2) {
                searchResults[idx] = [];
                target.innerHTML = '';
                return;
            }
            if (searchTimers[idx]) {
                clearTimeout(searchTimers[idx]);
            }
            target.innerHTML = '<div class="no-suggestions">Searching...</div>';
            searchTimers[idx] = setTimeout(async () => {
                try {
                    const resp = await fetch(`/api/search-players?q=${encodeURIComponent(trimmed)}`);
                    const data = await resp.json();
                    searchResults[idx] = data.results || [];
                    target.innerHTML = renderSearchResults(idx);
                } catch (e) {
                    target.innerHTML = '<div class="no-suggestions">Search failed</div>';
                }
            }, 250);
        }
        
        async function selectPlayer(idx, playerName) {
            selections[idx] = playerName;
            await fetch('/api/select', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({token_idx: idx, player_name: playerName})
            });
            renderTokens();
        }
        
        async function clearSelection(idx) {
            delete selections[idx];
            await fetch('/api/select', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({token_idx: idx, player_name: '__clear__'})
            });
            renderTokens();
        }
        
        function updateStats() {
            const total = tokensData.length;
            let matched = 0, noMatch = 0, pending = 0;
            
            tokensData.forEach((_, idx) => {
                const sel = selections[idx];
                if (sel === undefined) pending++;
                else if (sel === null) noMatch++;
                else matched++;
            });
            
            document.getElementById('total-tokens').textContent = total;
            document.getElementById('matched-tokens').textContent = matched;
            document.getElementById('no-match-tokens').textContent = noMatch;
            document.getElementById('pending-tokens').textContent = pending;
        }
        
        function proceedToStage3() {
            const matched = Object.values(selections).filter(s => s && s !== null).length;
            if (matched === 0) {
                const results = document.getElementById('llm-results');
                results.innerHTML = '<div style="color: #ff4444; text-align: center; padding: 50px;">Please select at least one player before proceeding.</div>';
                return;
            }
            document.querySelector('[data-stage="2"]').classList.add('completed');
            switchStage(3);
        }
        
        // LLM verification
        async function runLLMCheck() {
            const selectedPlayers = Object.values(selections).filter(s => s && s !== null);
            if (selectedPlayers.length === 0) {
                const results = document.getElementById('llm-results');
                results.innerHTML = '<div style="color: #ff4444; text-align: center; padding: 50px;">No players selected.</div>';
                return;
            }
            
            const btn = document.getElementById('btn-run-llm');
            const results = document.getElementById('llm-results');
            
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running LLM verification...';
            results.innerHTML = '<div style="text-align: center; color: #888; padding: 50px;">Verifying players with LLM...</div>';
            
            try {
                const response = await fetch('/api/run-llm', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        players: selectedPlayers,
                        question: document.getElementById('question-input').value
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'ok') {
                    llmResults = data.results || {};
                    renderLLMResults();
                    document.querySelector('[data-stage="3"]').classList.add('completed');
                } else {
                    results.innerHTML = '<div style="color: #ff4444; text-align: center; padding: 50px;">Error: ' + (data.error || 'Unknown error') + '</div>';
                }
            } catch (error) {
                results.innerHTML = '<div style="color: #ff4444; text-align: center; padding: 50px;">Network error: ' + error.message + '</div>';
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🚀 Run LLM Verification';
            }
        }
        
        function renderLLMResults() {
            const container = document.getElementById('llm-results');
            const summary = document.getElementById('llm-summary');
            container.innerHTML = '';
            let correct = 0;
            let wrong = 0;

            Object.entries(llmResults).forEach(([name, result]) => {
                const card = document.createElement('div');
                const answer = result.answer === true || result.answer === 'true' || result.answer === 'yes';
                const verdict = answer ? 'yes' : 'no';
                if (answer) correct++;
                else wrong++;
                
                card.className = `llm-card ${verdict}`;
                card.innerHTML = `
                    <div class="llm-player-name">${escapeHtml(name)}</div>
                    <div class="llm-verdict">
                        <span class="llm-verdict-icon">${answer ? '✅' : '❌'}</span>
                        <span class="llm-verdict-text">${answer ? 'CORRECT' : 'INCORRECT'}</span>
                    </div>
                    <div class="llm-justification">
                        ${escapeHtml(result.justification || 'No justification provided')}
                    </div>
                `;
                container.appendChild(card);
            });

            summary.style.display = 'block';
            summary.textContent = `✅ Correct: ${correct}   ❌ Incorrect: ${wrong}`;
        }
        
        // Export results
        async function exportResults() {
            const response = await fetch('/api/export');
            const data = await response.json();
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'quiz_verification_results.json';
            a.click();
        }
        
        // Utilities
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function escapeJs(text) {
            if (!text) return '';
            return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
        }
        
        // Initialize on load
        init();
    </script>
</body>
</html>
"""


def load_llm_client(path: str) -> Any:
    """Load LLM client from module:Class format."""
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ValueError("LLM client must be in module:Class format")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/init")
def api_init():
    return jsonify({
        "video_path": STATE["video_path"],
        "question": STATE["question"],
        "transcript": STATE["asr_result"].get("text", "") if STATE["asr_result"] else "",
    })


@app.route("/api/upload-video", methods=["POST"])
def api_upload_video():
    """Handle video file upload."""
    if 'video' not in request.files:
        return jsonify({"status": "error", "error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "error": "No file selected"}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time() * 1000))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        STATE["video_path"] = filepath
        STATE["uploaded_videos"][unique_filename] = filepath
        
        return jsonify({
            "status": "ok",
            "video_path": filepath,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/video")
def api_video():
    """Serve video file."""
    if STATE["video_path"] and Path(STATE["video_path"]).exists():
        return send_file(STATE["video_path"])
    return jsonify({"error": "Video not found"}), 404


@app.route("/api/run-asr", methods=["POST"])
def api_run_asr():
    """Run ASR on video segment."""
    data = request.json
    question = data.get("question", "")
    start_time = data.get("start_time", "0:00")
    end_time = data.get("end_time", "")
    
    STATE["question"] = question
    STATE["start_time"] = parse_timestamp(start_time)
    STATE["end_time"] = parse_timestamp(end_time) if end_time else 0
    
    try:
        # Import ASR functions
        from asr_steps.common import extract_audio, safe_transcribe, build_initial_prompt, select_prompt_names
        try:
            import whisper
        except ImportError:
            transcript = (
                os.environ.get("DUMMY_TRANSCRIPT_UI")
                or os.environ.get("DUMMY_TRANSCRIPT")
                or "Lionel Messi, Cristiano Ronaldo, Neymar."
            )
            STATE["asr_result"] = {"text": transcript, "segments": []}
            token_texts = re.findall(r"\w+", transcript, flags=re.UNICODE)
            tokens = [
                {
                    "token": t,
                    "segment_start": 0.0,
                    "segment_end": 0.0,
                    "probability": 1.0,
                    "avg_logprob": 0.0,
                }
                for t in token_texts
            ]
            STATE["tokens"] = tokens
            STATE["suggestions"] = _run_stage2_matching(tokens)
            STATE["selections"] = {}
            return jsonify(
                {
                    "status": "ok",
                    "transcript": transcript,
                    "tokens": len(token_texts),
                    "note": "Whisper not installed; using dummy transcript.",
                }
            )
        
        # Extract audio
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "clip.wav"
            extract_audio(
                STATE["video_path"],
                str(audio_path),
                start_time,
                end_time
            )
            
            # Load Whisper model
            model = whisper.load_model(STATE["whisper_model"])
            
            # Build prompt
            initial_prompt = None
            if STATE["player_db_path"] and Path(STATE["player_db_path"]).exists():
                known_names = select_prompt_names(
                    question,
                    Path(STATE["player_db_path"]),
                    Path(STATE["player_db_path"]),
                    1000,
                    False
                )
                initial_prompt = build_initial_prompt(known_names)
            
            # Transcribe
            result = safe_transcribe(
                model,
                str(audio_path),
                language="en",
                task="transcribe",
                initial_prompt=initial_prompt,
                temperature=0.4
            )
            
            transcript = result.get("text", "").strip()
            STATE["asr_result"] = result
            
            # Extract tokens
            token_texts = re.findall(r"\w+", transcript, flags=re.UNICODE)
            
            # Save tokens to temp CSV
            tokens_csv = Path(tmpdir) / "tokens.csv"
            with open(tokens_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["pass", "token", "segment_start", "segment_end", "probability", "avg_logprob", "no_speech_prob"])
                writer.writeheader()
                for token in token_texts:
                    writer.writerow({
                        "pass": 1,
                        "token": token,
                        "segment_start": 0,
                        "segment_end": 0,
                        "probability": 1.0,
                        "avg_logprob": 0,
                        "no_speech_prob": 0
                    })
            
            # Run stage2 matching to populate suggestions
            tokens = [
                {
                    "token": t,
                    "segment_start": 0.0,
                    "segment_end": 0.0,
                    "probability": 1.0,
                    "avg_logprob": 0.0,
                }
                for t in token_texts
            ]
            STATE["tokens"] = tokens
            STATE["suggestions"] = _run_stage2_matching(tokens)
            STATE["selections"] = {}
            
            return jsonify({
                "status": "ok",
                "transcript": transcript,
                "tokens": len(token_texts)
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


def parse_timestamp(ts: str) -> float:
    """Parse timestamp to seconds."""
    if not ts:
        return 0
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0]) if parts[0] else 0
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return 0


def _normalize_token(text: str) -> str:
    parts = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(parts).strip()


def _build_token_suggestions(tokens: List[Dict], matches: List[Dict]) -> Dict[int, List[Dict]]:
    """Aggregate match suggestions per token index."""
    token_suggestions: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    token_text_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        norm = _normalize_token(tok.get("token", ""))
        if norm:
            token_text_to_indices[norm].append(idx)

    for match in matches:
        pass_num = match.get("pass", 1)
        indices = match.get("token_indices", [])
        ngram = match.get("ngram", "")
        suggestions_list = match.get("suggestions", [])
        if not indices or len(indices) < 2:
            continue
        start_idx, end_idx = indices[0], indices[1]

        for suggestion in suggestions_list:
            name = suggestion.get("name", "")
            if not name:
                continue
            key = name.lower()

            if pass_num == 1:
                for idx in range(start_idx, end_idx + 1):
                    if idx < len(tokens):
                        if key not in token_suggestions[idx]:
                            token_suggestions[idx][key] = {**suggestion, "source_ngrams": [ngram]}
                        else:
                            existing = token_suggestions[idx][key]
                            if ngram not in existing.get("source_ngrams", []):
                                existing.setdefault("source_ngrams", []).append(ngram)
                            if (suggestion.get("career_score") or 0) > (existing.get("career_score") or 0):
                                existing.update(suggestion)
            else:
                for ng_token in ngram.split():
                    matched_indices = token_text_to_indices.get(ng_token, [])
                    for idx in matched_indices:
                        if key not in token_suggestions[idx]:
                            token_suggestions[idx][key] = {
                                **suggestion,
                                "source_ngrams": [ngram],
                                "source_passes": [pass_num],
                            }
                        else:
                            existing = token_suggestions[idx][key]
                            if ngram not in existing.get("source_ngrams", []):
                                existing.setdefault("source_ngrams", []).append(ngram)
                            if pass_num not in existing.get("source_passes", []):
                                existing.setdefault("source_passes", []).append(pass_num)
                            if (suggestion.get("career_score") or 0) > (existing.get("career_score") or 0):
                                existing.update(suggestion)

    result: Dict[int, List[Dict]] = {}
    for idx, suggestions_dict in token_suggestions.items():
        sorted_suggestions = sorted(
            suggestions_dict.values(),
            key=lambda s: (s.get("career_score") or 0, s.get("score") or 0),
            reverse=True,
        )
        result[idx] = sorted_suggestions
    return result


def _run_stage2_matching(tokens: List[Dict]) -> Dict[int, List[Dict]]:
    if not STATE["player_db_path"] or not Path(STATE["player_db_path"]).exists():
        return {}
    if STATE.get("players_by_name") and STATE.get("all_names"):
        players_by_name = STATE["players_by_name"]
        all_names = STATE["all_names"]
    else:
        players_by_name, all_names = load_players(Path(STATE["player_db_path"]))
    matches: List[Dict] = []
    search_cache: Dict[str, List[Dict]] = {}
    matches.extend(
        process_pass(
            1,
            tokens,
            players_by_name,
            all_names,
            STAGE2_MIN_GRAM,
            STAGE2_MAX_GRAM,
            STAGE2_FUZZY_THRESHOLD,
            STAGE2_MAX_SUGGESTIONS,
            search_cache,
            debug=False,
        )
    )
    return _build_token_suggestions(tokens, matches)


@app.route("/api/tokens")
def api_tokens():
    """Get tokens and suggestions."""
    tokens_with_suggestions = []
    for idx, token in enumerate(STATE["tokens"]):
        tokens_with_suggestions.append({
            **token,
            "suggestions": STATE["suggestions"].get(idx, []),
        })
    return jsonify({
        "tokens": tokens_with_suggestions,
        "selections": STATE["selections"]
    })


@app.route("/api/search-players")
def api_search_players():
    """Search players by query string."""
    query = (request.args.get("q") or "").strip()
    if not query or len(query) < 2:
        return jsonify({"results": []})

    if not STATE["player_db_path"] or not Path(STATE["player_db_path"]).exists():
        return jsonify({"results": []})

    if not STATE.get("players_by_name") or not STATE.get("all_names"):
        players_by_name, all_names = load_players(Path(STATE["player_db_path"]))
        STATE["players_by_name"] = players_by_name
        STATE["all_names"] = all_names
    else:
        players_by_name = STATE["players_by_name"]
        all_names = STATE["all_names"]

    norm_query = normalize(query)
    if not norm_query:
        return jsonify({"results": []})

    matches = fuzzy_match(norm_query, all_names, limit=12, threshold=60)
    results: List[Dict] = []
    seen = set()
    for matched_name, score in matches:
        for player in players_by_name.get(matched_name, [])[:3]:
            name = player.get("name") or player.get("full_name")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            career_score = player.get("_career_score") or 0.0
            results.append({
                "name": name,
                "match_type": "search",
                "score": score,
                "career_score": career_score,
                "player": {
                    "name": name,
                    "full_name": player.get("full_name"),
                    "nationality": player.get("nationality"),
                    "position": player.get("position"),
                    "current_club": player.get("current_club") or player.get("club"),
                    "career_score": career_score,
                },
            })

    results.sort(key=lambda s: (s.get("career_score") or 0, s.get("score") or 0), reverse=True)
    return jsonify({"results": results})


@app.route("/api/set-transcript", methods=["POST"])
def api_set_transcript():
    """Override transcript and regenerate tokens."""
    data = request.json or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"status": "error", "error": "Transcript is required"}), 400
    token_texts = re.findall(r"\w+", transcript, flags=re.UNICODE)
    tokens = [
        {
            "token": t,
            "segment_start": 0.0,
            "segment_end": 0.0,
            "probability": 1.0,
            "avg_logprob": 0.0,
        }
        for t in token_texts
    ]
    STATE["asr_result"] = {"text": transcript, "segments": []}
    STATE["tokens"] = tokens
    STATE["suggestions"] = _run_stage2_matching(tokens)
    STATE["selections"] = {}
    STATE["llm_results"] = {}
    return jsonify({"status": "ok", "tokens": len(token_texts)})


@app.route("/api/select", methods=["POST"])
def api_select():
    """Select a player for a token."""
    data = request.json
    token_idx = data.get("token_idx")
    player_name = data.get("player_name")
    
    if player_name == "__clear__":
        STATE["selections"].pop(token_idx, None)
    else:
        STATE["selections"][token_idx] = player_name
    
    return jsonify({"status": "ok"})


@app.route("/api/run-llm", methods=["POST"])
def api_run_llm():
    """Run LLM verification on selected players."""
    data = request.json
    players = data.get("players", [])
    question = data.get("question", "")
    
    results = {}
    if STATE["llm_client"]:
        for player_name in players:
            prompt = (
                f"Answer the question for the single player below. "
                f"Return strict JSON: {{\"answer\": true|false, \"justification\": \"...\"}}. "
                f"Include years/dates if relevant.\n"
                f"Question: {question}\n"
                f"Player: {player_name}\n"
            )
            try:
                response = STATE["llm_client"].ask(prompt)
                try:
                    result = json.loads(response)
                except Exception:
                    result = {
                        "answer": "true" in response.lower() or "yes" in response.lower(),
                        "justification": response,
                    }
                results[player_name] = result
            except Exception as e:
                results[player_name] = {
                    "answer": False,
                    "justification": f"Error: {str(e)}",
                }
    else:
        try:
            all_valid, invalid_names, reasoning = verify_with_llm(
                players,
                question,
                player_db=STATE.get("player_db"),
                llm_provider=STATE.get("llm_provider") or "gemini",
                model=STATE.get("llm_model"),
            )
            invalid_set = {n.lower() for n in invalid_names}
            for player_name in players:
                is_valid = player_name.lower() not in invalid_set
                results[player_name] = {
                    "answer": is_valid,
                    "justification": (
                        f"LLM says this player {'satisfies' if is_valid else 'does not satisfy'} the question. "
                        f"Reasoning: {reasoning}"
                    ),
                }
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    
    STATE["llm_results"] = results
    return jsonify({"status": "ok", "results": results})


@app.route("/api/export")
def api_export():
    """Export all results."""
    return jsonify({
        "question": STATE["question"],
        "video_path": STATE["video_path"],
        "start_time": STATE["start_time"],
        "end_time": STATE["end_time"],
        "transcript": STATE["asr_result"].get("text", "") if STATE["asr_result"] else "",
        "tokens": STATE["tokens"],
        "selections": STATE["selections"],
        "llm_results": STATE["llm_results"]
    })


def main():
    parser = argparse.ArgumentParser(description="Integrated UI for all 3 stages with video upload")
    parser.add_argument("--player-db", default="data/players_enriched.jsonl", help="Player database JSONL")
    parser.add_argument("--question", help="Initial question")
    parser.add_argument("--whisper-model", default="large", help="Whisper model size")
    parser.add_argument("--llm-client", help="LLM client module:Class")
    parser.add_argument("--llm-provider", default="gemini", choices=["gemini", "openai", "ollama", "anthropic"], help="LLM provider for stage 3")
    parser.add_argument("--llm-model", help="LLM model name (provider-specific)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--upload-dir", help="Directory to store uploaded videos")
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv()
    
    STATE["player_db_path"] = args.player_db
    STATE["question"] = args.question or ""
    STATE["whisper_model"] = args.whisper_model
    
    if args.upload_dir:
        app.config['UPLOAD_FOLDER'] = args.upload_dir
        Path(args.upload_dir).mkdir(parents=True, exist_ok=True)
    
    if args.llm_client:
        try:
            STATE["llm_client"] = load_llm_client(args.llm_client)
            print(f"✓ Loaded LLM client: {args.llm_client}")
        except Exception as e:
            print(f"⚠ Warning: Could not load LLM client: {e}")

    STATE["llm_provider"] = args.llm_provider
    STATE["llm_model"] = args.llm_model
    if args.player_db and Path(args.player_db).exists():
        STATE["player_db"] = load_player_database(args.player_db)
        players_by_name, all_names = load_players(Path(args.player_db))
        STATE["players_by_name"] = players_by_name
        STATE["all_names"] = all_names
    
    print(f"\n🚀 Starting integrated UI at http://{args.host}:{args.port}")
    print(f"   Player DB: {args.player_db}")
    print(f"   Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("   📹 Upload your video through the web interface")
    print("   Press Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
