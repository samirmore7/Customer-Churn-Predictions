import os
import pickle
import traceback
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ==============================================================================
# BASE DIRECTORY & MODEL INITIALIZATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ADABoost.pkl")

MODEL = None
LOAD_ERROR = None

try:
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size < 1000:
            with open(MODEL_PATH, "r") as f:
                content = f.read(200)
            if "version https://git-lfs" in content:
                LOAD_ERROR = (
                    f"GIT LFS POINTER DETECTED: File size is only {file_size} bytes. "
                    f"Re-upload the real binary file via GitHub web UI."
                )
        
        if not LOAD_ERROR:
            with open(MODEL_PATH, "rb") as f:
                MODEL = pickle.load(f)
    else:
        LOAD_ERROR = f"Model file not found at path '{MODEL_PATH}'. Files in directory: {os.listdir(BASE_DIR)}"
except Exception as e:
    LOAD_ERROR = f"PICKLE UNPICKLING FAILED: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"

# ==============================================================================
# CATEGORICAL FEATURE ENCODING MAPPINGS
# ==============================================================================
GENDER_MAP = {
    "Male": 0,
    "Female": 1
}

SUBSCRIPTION_MAP = {
    "Basic": 0,
    "Standard": 1,
    "Premium": 2
}

CONTRACT_MAP = {
    "Monthly": 0,
    "Quarterly": 1,
    "Annual": 2
}

# ==============================================================================
# FULL UNCOMPRESSED PREMIUM HTML / CSS / JS TEMPLATE
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="obsidian">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Churn Analytics Studio - Enterprise Edition</title>
    
    <!-- Modern Google Fonts Typography Stack -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,600&display=swap" rel="stylesheet">
    
    <!-- FontAwesome Icon Suite -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Chart.js Visualization Library -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>

    <style>
        /* ==========================================================================
           ROOT DESIGN VARIABLES & THEME DEFINITIONS
           ========================================================================== */
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-tertiary: #1f2937;
            --bg-glass: rgba(17, 24, 39, 0.85);
            --bg-glass-card: rgba(31, 41, 55, 0.45);
            --bg-glass-input: rgba(0, 0, 0, 0.25);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(99, 102, 241, 0.3);
            --accent-glow: #6366f1;
            --accent-secondary: #a855f7;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-gradient-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --text-subtle: #6b7280;
            --card-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
            --glass-blur: blur(16px);
            --transition-speed: 0.3s;
        }

        /* --- THEME 1: OBSIDIAN CYBER --- */
        [data-theme="obsidian"] {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-glass: rgba(17, 24, 39, 0.85);
            --bg-glass-card: rgba(31, 41, 55, 0.45);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-glow: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --card-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
        }

        /* --- THEME 2: AMETHYST GLASS --- */
        [data-theme="amethyst"] {
            --bg-primary: #0d0914;
            --bg-secondary: #191224;
            --bg-glass: rgba(25, 18, 36, 0.85);
            --bg-glass-card: rgba(42, 31, 61, 0.45);
            --border-color: rgba(216, 180, 254, 0.12);
            --accent-glow: #c084fc;
            --accent-gradient: linear-gradient(135deg, #c084fc 0%, #e879f9 100%);
            --text-main: #faf5ff;
            --text-muted: #c0a9d4;
            --card-shadow: 0 20px 40px rgba(18, 5, 30, 0.55);
        }

        /* --- THEME 3: EMERALD EXECUTIVE --- */
        [data-theme="emerald"] {
            --bg-primary: #061412;
            --bg-secondary: #0b2420;
            --bg-glass: rgba(11, 36, 32, 0.85);
            --bg-glass-card: rgba(18, 56, 50, 0.45);
            --border-color: rgba(52, 211, 153, 0.12);
            --accent-glow: #10b981;
            --accent-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --text-main: #ecfdf5;
            --text-muted: #86efac;
            --card-shadow: 0 20px 40px rgba(2, 20, 15, 0.55);
        }

        /* --- THEME 4: MIDNIGHT GOLD --- */
        [data-theme="gold"] {
            --bg-primary: #120f0a;
            --bg-secondary: #211c13;
            --bg-glass: rgba(33, 28, 19, 0.85);
            --bg-glass-card: rgba(51, 43, 29, 0.45);
            --border-color: rgba(251, 191, 36, 0.15);
            --accent-glow: #f59e0b;
            --accent-gradient: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            --text-main: #fffbeb;
            --text-muted: #fcd34d;
            --card-shadow: 0 20px 40px rgba(20, 15, 5, 0.55);
        }

        /* --- THEME 5: ROSE GOLD LUXE --- */
        [data-theme="rosegold"] {
            --bg-primary: #170a0f;
            --bg-secondary: #27121a;
            --bg-glass: rgba(39, 18, 26, 0.85);
            --bg-glass-card: rgba(58, 27, 39, 0.45);
            --border-color: rgba(251, 113, 133, 0.15);
            --accent-glow: #fb7185;
            --accent-gradient: linear-gradient(135deg, #fb7185 0%, #e11d48 100%);
            --text-main: #fff1f2;
            --text-muted: #fecdd3;
            --card-shadow: 0 20px 40px rgba(25, 5, 12, 0.55);
        }

        /* --- THEME 6: ELECTRIC CYAN --- */
        [data-theme="cyan"] {
            --bg-primary: #04131a;
            --bg-secondary: #08232f;
            --bg-glass: rgba(8, 35, 47, 0.85);
            --bg-glass-card: rgba(14, 55, 73, 0.45);
            --border-color: rgba(6, 182, 212, 0.15);
            --accent-glow: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
            --text-main: #ecfeff;
            --text-muted: #a5f3fc;
            --card-shadow: 0 20px 40px rgba(2, 25, 35, 0.55);
        }

        /* --- THEME 7: SUNSET CRIMSON --- */
        [data-theme="crimson"] {
            --bg-primary: #1a0909;
            --bg-secondary: #2c0f0f;
            --bg-glass: rgba(44, 15, 15, 0.85);
            --bg-glass-card: rgba(66, 23, 23, 0.45);
            --border-color: rgba(248, 113, 113, 0.15);
            --accent-glow: #ef4444;
            --accent-gradient: linear-gradient(135deg, #f97316 0%, #ef4444 100%);
            --text-main: #fef2f2;
            --text-muted: #fca5a5;
            --card-shadow: 0 20px 40px rgba(30, 5, 5, 0.55);
        }

        /* --- THEME 8: TITANIUM PLATINUM --- */
        [data-theme="platinum"] {
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-glass: rgba(255, 255, 255, 0.9);
            --bg-glass-card: rgba(255, 255, 255, 0.85);
            --border-color: rgba(0, 0, 0, 0.08);
            --accent-glow: #2563eb;
            --accent-gradient: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --card-shadow: 0 20px 40px rgba(0, 0, 0, 0.06);
        }

        /* ==========================================================================
           GLOBAL RESET & BASE STYLES
           ========================================================================== */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            transition: background-color var(--transition-speed) ease, 
                        border-color var(--transition-speed) ease, 
                        color var(--transition-speed) ease,
                        box-shadow var(--transition-speed) ease;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.06) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.06) 0%, transparent 40%);
            background-attachment: fixed;
        }

        /* ==========================================================================
           HEADER ARCHITECTURE
           ========================================================================== */
        header {
            padding: 1.1rem 2.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            background: var(--bg-glass);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: 1.35rem;
            letter-spacing: -0.025em;
            color: var(--text-main);
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: var(--accent-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 1.15rem;
            box-shadow: 0 0 20px var(--accent-glow);
        }

        .theme-selector {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            background: var(--bg-glass-card);
            padding: 0.4rem 0.75rem;
            border-radius: 30px;
            border: 1px solid var(--border-color);
            backdrop-filter: var(--glass-blur);
        }

        .theme-btn {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
            transition: transform 0.2s ease, border-color 0.2s ease;
            position: relative;
        }

        .theme-btn:hover {
            transform: scale(1.25);
        }

        .theme-btn.active {
            border-color: #ffffff;
            transform: scale(1.15);
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }

        .theme-obsidian { background: #6366f1; }
        .theme-amethyst { background: #c084fc; }
        .theme-emerald { background: #10b981; }
        .theme-gold { background: #f59e0b; }
        .theme-rosegold { background: #fb7185; }
        .theme-cyan { background: #06b6d4; }
        .theme-crimson { background: #ef4444; }
        .theme-platinum { background: #64748b; }

        /* ==========================================================================
           GRID SYSTEM & CARD LAYOUTS
           ========================================================================== */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.35rem;
            max-width: 1550px;
            margin: 1.75rem auto;
            padding: 0 1.75rem;
            width: 100%;
        }

        .span-1 { grid-column: span 1; }
        .span-2 { grid-column: span 2; }
        .span-3 { grid-column: span 3; }
        .span-4 { grid-column: span 4; }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .span-1, .span-2, .span-3, .span-4 {
                grid-column: span 1;
            }
        }

        .glass-card {
            background: var(--bg-glass-card);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1.5rem 1.65rem;
            box-shadow: var(--card-shadow);
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-gradient);
            opacity: 0.7;
        }

        .section-label {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.025rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            letter-spacing: -0.01em;
            color: var(--text-main);
        }

        .subtext-label {
            font-size: 0.725rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }

        /* ==========================================================================
           KPI DASHBOARD WIDGETS
           ========================================================================== */
        .kpi-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 125px;
        }

        .kpi-value {
            font-family: 'Inter', sans-serif;
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0.35rem 0;
            color: var(--text-main);
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .kpi-subtext {
            font-size: 0.775rem;
            font-weight: 600;
            color: #34d399;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        /* ==========================================================================
           FORM INPUT CONTROLS & SELECTION
           ========================================================================== */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 0.5rem;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .input-group label {
            font-size: 0.715rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .input-control {
            background: var(--bg-glass-input);
            border: 1px solid var(--border-color);
            border-radius: 9px;
            padding: 0.65rem 0.85rem;
            color: var(--text-main);
            font-size: 0.885rem;
            font-weight: 500;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-control:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18);
            background: rgba(0, 0, 0, 0.38);
        }

        select.input-control option {
            background-color: var(--bg-secondary);
            color: var(--text-main);
        }

        /* ==========================================================================
           BUTTONS & ACTION HANDLERS
           ========================================================================== */
        .btn-group {
            grid-column: span 2;
            display: flex;
            gap: 0.9rem;
            margin-top: 0.65rem;
        }

        @media (max-width: 640px) {
            .btn-group {
                grid-column: span 1;
                flex-direction: column;
            }
        }

        .btn-premium {
            flex: 2;
            background: var(--accent-gradient);
            color: #ffffff;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.01em;
            padding: 0.8rem 1.25rem;
            border: none;
            border-radius: 11px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.55rem;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.28);
            transition: all 0.25s ease;
        }

        .btn-premium:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 26px rgba(99, 102, 241, 0.42);
        }

        .btn-secondary {
            flex: 1;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.8rem;
            border-radius: 11px;
            cursor: pointer;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.09);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* ==========================================================================
           RESULTS BADGE & RADAR DISPLAY
           ========================================================================== */
        .result-box {
            text-align: center;
            padding: 1.15rem;
            background: rgba(0, 0, 0, 0.22);
            border-radius: 14px;
            border: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }

        .badge-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 1rem;
            border-radius: 30px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
            letter-spacing: 0.02em;
        }

        .badge-pos {
            background: rgba(16, 185, 129, 0.16);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.32);
        }

        .badge-neg {
            background: rgba(239, 68, 68, 0.16);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.32);
        }

        .chart-container {
            position: relative;
            height: 230px;
            width: 100%;
        }

        /* ==========================================================================
           HISTORY ANALYTICS TABLE
           ========================================================================== */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5rem;
            font-size: 0.835rem;
        }

        th, td {
            padding: 0.7rem 0.9rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            color: var(--text-muted);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.685rem;
            letter-spacing: 0.07em;
        }

        footer {
            text-align: center;
            padding: 1.5rem;
            font-size: 0.825rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            margin-top: auto;
            backdrop-filter: var(--glass-blur);
            background: var(--bg-glass);
        }

        .spinner {
            display: none;
            width: 17px;
            height: 17px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <!-- ======================================================================
         APPLICATION NAVIGATION HEADER
         ====================================================================== -->
    <header>
        <div class="logo">
            <div class="logo-icon"><i class="fa-solid fa-chart-pie"></i></div>
            <span>AI Churn <span style="font-weight: 400; opacity: 0.8;">Analytics Studio</span></span>
        </div>
        
        <div class="theme-selector">
            <span class="subtext-label" style="margin-right: 0.3rem;"><i class="fa-solid fa-palette"></i> Themes</span>
            <button class="theme-btn theme-obsidian active" onclick="setTheme('obsidian')" title="Obsidian Cyber"></button>
            <button class="theme-btn theme-amethyst" onclick="setTheme('amethyst')" title="Amethyst Glass"></button>
            <button class="theme-btn theme-emerald" onclick="setTheme('emerald')" title="Emerald Executive"></button>
            <button class="theme-btn theme-gold" onclick="setTheme('gold')" title="Midnight Gold"></button>
            <button class="theme-btn theme-rosegold" onclick="setTheme('rosegold')" title="Rose Gold Luxe"></button>
            <button class="theme-btn theme-cyan" onclick="setTheme('cyan')" title="Electric Cyan"></button>
            <button class="theme-btn theme-crimson" onclick="setTheme('crimson')" title="Sunset Crimson"></button>
            <button class="theme-btn theme-platinum" onclick="setTheme('platinum')" title="Titanium Platinum"></button>
        </div>
    </header>

    <!-- ======================================================================
         MAIN DASHBOARD LAYOUT GRID
         ====================================================================== -->
    <div class="dashboard-grid">
        
        <!-- KPI CARD 1 -->
        <div class="glass-card kpi-card span-1">
            <div class="subtext-label">Total Evaluated</div>
            <div class="kpi-value" id="kpiTotal">1,248</div>
            <div class="kpi-subtext"><i class="fa-solid fa-arrow-up"></i> +14% this month</div>
        </div>

        <!-- KPI CARD 2 -->
        <div class="glass-card kpi-card span-1">
            <div class="subtext-label">Positive Rate (Class 1)</div>
            <div class="kpi-value" id="kpiRate">68.4%</div>
            <div class="kpi-subtext" style="color: #60a5fa;"><i class="fa-solid fa-users"></i> Optimal distribution</div>
        </div>

        <!-- KPI CARD 3 -->
        <div class="glass-card kpi-card span-1">
            <div class="subtext-label">Avg User Spend</div>
            <div class="kpi-value" id="kpiSpend">$850.50</div>
            <div class="kpi-subtext"><i class="fa-solid fa-dollar-sign"></i> +5.2% vs average</div>
        </div>

        <!-- KPI CARD 4 -->
        <div class="glass-card kpi-card span-1">
            <div class="subtext-label">Model Status</div>
            <div class="kpi-value" style="color: #34d399;">Active</div>
            <div class="kpi-subtext" style="color: #a7f3d0;"><i class="fa-solid fa-shield-halved"></i> AdaBoost Classifier</div>
        </div>

        <!-- FEATURE FORM CARD -->
        <div class="glass-card span-2">
            <div style="margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center;">
                <span class="section-label"><i class="fa-solid fa-sliders" style="color:var(--accent-glow)"></i> Input Features</span>
                <button class="btn-secondary" style="padding: 0.35rem 0.75rem; font-size: 0.75rem;" onclick="loadPresetSample()">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Preset High-Value
                </button>
            </div>

            <form id="predictionForm" onsubmit="handlePredict(event)">
                <div class="form-grid">
                    <div class="input-group">
                        <label>Age (Years)</label>
                        <input type="number" id="age" class="input-control" value="34" min="18" max="100" required>
                    </div>

                    <div class="input-group">
                        <label>Gender</label>
                        <select id="gender" class="input-control" required>
                            <option value="Male" selected>Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Tenure (Months)</label>
                        <input type="number" id="tenure" class="input-control" value="24" min="0" max="120" required>
                    </div>

                    <div class="input-group">
                        <label>Usage Frequency</label>
                        <input type="number" id="usage_frequency" class="input-control" value="18" min="0" max="100" required>
                    </div>

                    <div class="input-group">
                        <label>Support Calls</label>
                        <input type="number" id="support_calls" class="input-control" value="2" min="0" max="50" required>
                    </div>

                    <div class="input-group">
                        <label>Payment Delay (Days)</label>
                        <input type="number" id="payment_delay" class="input-control" value="1" min="0" max="60" required>
                    </div>

                    <div class="input-group">
                        <label>Subscription Type</label>
                        <select id="subscription_type" class="input-control" required>
                            <option value="Basic">Basic</option>
                            <option value="Standard" selected>Standard</option>
                            <option value="Premium">Premium</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Contract Length</label>
                        <select id="contract_length" class="input-control" required>
                            <option value="Monthly">Monthly</option>
                            <option value="Quarterly">Quarterly</option>
                            <option value="Annual" selected>Annual</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Total Spend ($)</label>
                        <input type="number" step="0.01" id="total_spend" class="input-control" value="850.50" min="0" required>
                    </div>

                    <div class="input-group">
                        <label>Last Interaction (Days)</label>
                        <input type="number" id="last_interaction" class="input-control" value="5" min="0" max="365" required>
                    </div>

                    <div class="btn-group">
                        <button type="submit" class="btn-premium">
                            <span class="spinner" id="spinner"></span>
                            <i class="fa-solid fa-bolt" id="btnIcon"></i>
                            <span>Execute Prediction</span>
                        </button>
                        <button type="button" class="btn-secondary" onclick="resetForm()">
                            <i class="fa-solid fa-rotate-left"></i> Reset
                        </button>
                    </div>
                </div>
            </form>
        </div>

        <!-- RADAR OUTPUT DISPLAY CARD -->
        <div class="glass-card span-2">
            <div style="margin-bottom:0.85rem;" class="section-label">
                <i class="fa-solid fa-chart-radar" style="color:var(--accent-glow)"></i> Input Feature Radar Profile
            </div>
            
            <div class="result-box">
                <div class="badge-status badge-pos" id="resultBadge">
                    <i class="fa-solid fa-check-circle"></i> <span id="predictionResult">Ready for Analysis</span>
                </div>
                <div style="font-size: 0.775rem; color: var(--text-muted); margin-top: 0.35rem;" id="resultDesc">
                    Configure feature inputs on the left and click Execute Prediction to evaluate model output.
                </div>
            </div>

            <div class="chart-container">
                <canvas id="radarChart"></canvas>
            </div>
        </div>

        <!-- HISTORY LOG TABLE -->
        <div class="glass-card span-4">
            <div style="margin-bottom:0.85rem; display:flex; justify-content:space-between; align-items:center;">
                <span class="section-label"><i class="fa-solid fa-list-check" style="color:var(--accent-glow)"></i> AI Churn Analytics History Log</span>
            </div>
            <div style="overflow-x: auto;">
                <table id="logsTable">
                    <thead>
                        <tr>
                            <th>Gender</th>
                            <th>Subscription</th>
                            <th>Contract</th>
                            <th>Tenure</th>
                            <th>Total Spend</th>
                            <th>Calls / Delay</th>
                            <th>Probability</th>
                            <th>Prediction Output</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Female</td>
                            <td>Premium</td>
                            <td>Annual</td>
                            <td>36 mos</td>
                            <td>$1,450.00</td>
                            <td>0 / 0</td>
                            <td>0.942</td>
                            <td><span class="badge-status badge-pos" style="font-size:0.7rem; padding: 0.15rem 0.5rem; margin-bottom:0;">Class 1</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <footer>
        AI Churn Analytics Studio &bull; Flask & Scikit-Learn Enterprise Deployment
    </footer>

    <!-- ======================================================================
         INTERACTIVE JAVASCRIPT LOGIC
         ====================================================================== -->
    <script>
        let radarChart = null;

        function setTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.querySelector(`.theme-${themeName}`);
            if (activeBtn) activeBtn.classList.add('active');
            updateChartTheme();
        }

        function loadPresetSample() {
            document.getElementById('age').value = 45;
            document.getElementById('gender').value = 'Female';
            document.getElementById('tenure').value = 36;
            document.getElementById('usage_frequency').value = 28;
            document.getElementById('support_calls').value = 0;
            document.getElementById('payment_delay').value = 0;
            document.getElementById('subscription_type').value = 'Premium';
            document.getElementById('contract_length').value = 'Annual';
            document.getElementById('total_spend').value = 1450.00;
            document.getElementById('last_interaction').value = 2;
        }

        function resetForm() {
            document.getElementById('predictionForm').reset();
        }

        function initChart() {
            try {
                const ctx = document.getElementById('radarChart');
                if (!ctx || typeof Chart === 'undefined') return;

                radarChart = new Chart(ctx.getContext('2d'), {
                    type: 'radar',
                    data: {
                        labels: ['Tenure', 'Usage', 'Calls', 'Delay', 'Spend Scale', 'Interaction'],
                        datasets: [{
                            label: 'Feature Scale',
                            data: [24, 18, 2, 1, 85, 5],
                            backgroundColor: 'rgba(99, 102, 241, 0.25)',
                            borderColor: '#6366f1',
                            borderWidth: 2,
                            pointBackgroundColor: '#6366f1'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255,255,255,0.1)' },
                                grid: { color: 'rgba(255,255,255,0.1)' },
                                pointLabels: { color: '#9ca3af', font: { size: 10, family: 'Inter' } },
                                ticks: { display: false }
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            } catch (err) {
                console.warn('Chart initialization skipped:', err);
            }
        }

        function updateChartTheme() {
            if (!radarChart) return;
            const style = getComputedStyle(document.documentElement);
            const accent = style.getPropertyValue('--accent-glow').trim() || '#6366f1';
            radarChart.data.datasets[0].backgroundColor = accent + '40';
            radarChart.data.datasets[0].borderColor = accent;
            radarChart.data.datasets[0].pointBackgroundColor = accent;
            radarChart.update();
        }

        async function handlePredict(event) {
            event.preventDefault();

            const spinner = document.getElementById('spinner');
            const btnIcon = document.getElementById('btnIcon');
            if (spinner) spinner.style.display = 'inline-block';
            if (btnIcon) btnIcon.style.display = 'none';

            const payload = {
                Age: parseFloat(document.getElementById('age').value),
                Gender: document.getElementById('gender').value,
                Tenure: parseFloat(document.getElementById('tenure').value),
                Usage_Frequency: parseFloat(document.getElementById('usage_frequency').value),
                Support_Calls: parseFloat(document.getElementById('support_calls').value),
                Payment_Delay: parseFloat(document.getElementById('payment_delay').value),
                Subscription_Type: document.getElementById('subscription_type').value,
                Contract_Length: document.getElementById('contract_length').value,
                Total_Spend: parseFloat(document.getElementById('total_spend').value),
                Last_Interaction: parseFloat(document.getElementById('last_interaction').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (data.status === 'success') {
                    const badge = document.getElementById('resultBadge');
                    const resTitle = document.getElementById('predictionResult');
                    const resDesc = document.getElementById('resultDesc');

                    resTitle.innerText = data.prediction_label;
                    
                    if (data.prediction === 1) {
                        badge.className = 'badge-status badge-pos';
                        badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${data.prediction_label}`;
                        resDesc.innerText = `High probability positive output (${(data.probability * 100).toFixed(1)}% confidence).`;
                    } else {
                        badge.className = 'badge-status badge-neg';
                        badge.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${data.prediction_label}`;
                        resDesc.innerText = `Standard/Negative outcome identified (${(data.probability * 100).toFixed(1)}% confidence).`;
                    }

                    const tbody = document.querySelector('#logsTable tbody');
                    if (tbody) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${payload.Gender}</td>
                            <td>${payload.Subscription_Type}</td>
                            <td>${payload.Contract_Length}</td>
                            <td>${payload.Tenure} mos</td>
                            <td>$${payload.Total_Spend.toFixed(2)}</td>
                            <td>${payload.Support_Calls} / ${payload.Payment_Delay}d</td>
                            <td>${data.probability.toFixed(3)}</td>
                            <td><span class="badge-status ${data.prediction === 1 ? 'badge-pos' : 'badge-neg'}" style="font-size:0.7rem; padding: 0.15rem 0.5rem; margin-bottom:0;">${data.prediction_label}</span></td>
                        `;
                        tbody.insertBefore(row, tbody.firstChild);
                    }

                    if (radarChart) {
                        radarChart.data.datasets[0].data = [
                            payload.Tenure, payload.Usage_Frequency, payload.Support_Calls, 
                            payload.Payment_Delay, Math.min(payload.Total_Spend / 10, 100), payload.Last_Interaction
                        ];
                        radarChart.update();
                    }
                } else {
                    alert('Prediction Failed:\n\n' + data.message);
                }
            } catch (err) {
                alert('Network Connection Error: ' + err.message);
            } finally {
                if (spinner) spinner.style.display = 'none';
                if (btnIcon) btnIcon.style.display = 'inline-block';
            }
        }

        window.addEventListener('DOMContentLoaded', function() {
            initChart();
            updateChartTheme();
        });
    </script>
</body>
</html>
"""

# ==============================================================================
# ROUTE CONTROLLERS
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/debug")
def debug():
    if LOAD_ERROR:
        return f"<h3>Model Load Diagnostic Error:</h3><pre>{LOAD_ERROR}</pre>"
    return "<h3>Model Loaded Successfully!</h3>"

@app.route("/predict", methods=["POST"])
def predict():
    if LOAD_ERROR or MODEL is None:
        return jsonify({"status": "error", "message": f"Model failed to load at startup: {LOAD_ERROR}"}), 500

    try:
        data = request.get_json(force=True)

        gender_num = GENDER_MAP.get(data.get("Gender", "Male"), 0)
        sub_num = SUBSCRIPTION_MAP.get(data.get("Subscription_Type", "Basic"), 0)
        contract_num = CONTRACT_MAP.get(data.get("Contract_Length", "Monthly"), 0)

        features = np.array([[
            float(data.get("Age", 0)),
            gender_num,
            float(data.get("Tenure", 0)),
            float(data.get("Usage_Frequency", 0)),
            float(data.get("Support_Calls", 0)),
            float(data.get("Payment_Delay", 0)),
            sub_num,
            contract_num,
            float(data.get("Total_Spend", 0)),
            float(data.get("Last_Interaction", 0))
        ]])

        prediction = int(MODEL.predict(features)[0])
        probability = float(MODEL.predict_proba(features)[0][prediction]) if hasattr(MODEL, "predict_proba") else 1.0

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "prediction_label": "Class 1" if prediction == 1 else "Class 0",
            "probability": probability
        })

    except Exception as e:
        error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return jsonify({"status": "error", "message": error_msg}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
