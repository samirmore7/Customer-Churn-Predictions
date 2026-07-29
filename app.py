import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ADABoost.pkl")
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Categorical mappings matching model training configuration
CATEGORICAL_MAPPINGS = {
    "Gender": {"Female": 0, "Male": 1},
    "Subscription Type": {"Basic": 0, "Standard": 1, "Premium": 2},
    "Contract Length": {"Monthly": 0, "Quarterly": 1, "Annual": 2}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark-glass">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Churn Prediction Studio</title>
    
    <!-- Google Fonts & FontAwesome Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        /* Theme Variables */
        :root {
            /* Theme 1: Dark Glass (Default) */
            --bg-base: #060913;
            --bg-surface: rgba(15, 23, 42, 0.65);
            --bg-card: rgba(30, 41, 59, 0.5);
            --border-card: rgba(255, 255, 255, 0.08);
            --border-focus: rgba(99, 102, 241, 0.5);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --accent-glow: rgba(99, 102, 241, 0.4);
            --btn-shadow: 0 0 25px rgba(99, 102, 241, 0.5);
        }

        [data-theme="emerald"] {
            /* Theme 2: Emerald Luxury */
            --bg-base: #021a12;
            --bg-surface: rgba(6, 44, 32, 0.7);
            --bg-card: rgba(11, 61, 44, 0.5);
            --border-card: rgba(52, 211, 153, 0.15);
            --border-focus: rgba(16, 185, 129, 0.6);
            --text-main: #f0fdf4;
            --text-muted: #86efac;
            --accent-gradient: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
            --accent-glow: rgba(16, 185, 129, 0.4);
            --btn-shadow: 0 0 25px rgba(16, 185, 129, 0.5);
        }

        [data-theme="cyberpunk"] {
            /* Theme 3: Cyberpunk Neon */
            --bg-base: #0f051d;
            --bg-surface: rgba(29, 9, 54, 0.75);
            --bg-card: rgba(49, 14, 89, 0.5);
            --border-card: rgba(236, 72, 153, 0.2);
            --border-focus: rgba(244, 114, 182, 0.7);
            --text-main: #fff1f2;
            --text-muted: #f472b6;
            --accent-gradient: linear-gradient(135deg, #ff007f 0%, #7928ca 50%, #00dfd8 100%);
            --accent-glow: rgba(255, 0, 127, 0.5);
            --btn-shadow: 0 0 25px rgba(255, 0, 127, 0.6);
        }

        [data-theme="light-minimal"] {
            /* Theme 4: Premium Light */
            --bg-base: #f1f5f9;
            --bg-surface: rgba(255, 255, 255, 0.85);
            --bg-card: rgba(255, 255, 255, 0.65);
            --border-card: rgba(0, 0, 0, 0.08);
            --border-focus: rgba(79, 70, 229, 0.4);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #d946ef 100%);
            --accent-glow: rgba(79, 70, 229, 0.25);
            --btn-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background 0.4s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.4s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Ambient Glow Backgrounds */
        .ambient-glow-1, .ambient-glow-2 {
            position: absolute;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: var(--accent-gradient);
            filter: blur(140px);
            opacity: 0.25;
            z-index: -1;
            animation: pulse-glow 8s ease-in-out infinite alternate;
        }
        .ambient-glow-1 { top: -100px; left: -100px; }
        .ambient-glow-2 { bottom: -100px; right: -100px; }

        @keyframes pulse-glow {
            0% { transform: scale(1) translate(0, 0); opacity: 0.2; }
            100% { transform: scale(1.2) translate(30px, 30px); opacity: 0.35; }
        }

        .container {
            max-width: 1240px;
            margin: 0 auto;
        }

        /* Glassmorphism Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2rem;
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid var(--border-card);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }

        .brand-box {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-icon {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            background: var(--accent-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 1.3rem;
            box-shadow: var(--btn-shadow);
        }

        .brand-title h1 {
            font-size: 1.4rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .brand-title p {
            font-size: 0.78rem;
            color: var(--text-muted);
        }

        /* Controls Area & Theme Selector */
        .controls-area {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .theme-select-wrapper {
            position: relative;
        }

        .theme-select {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-main);
            padding: 0.65rem 1.2rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .theme-select:hover {
            border-color: var(--border-focus);
            box-shadow: 0 0 15px var(--accent-glow);
        }

        /* Dashboard Main Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        /* Glass Cards */
        .glass-card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid var(--border-card);
            padding: 2rem;
            box-shadow: 0 12px 40px rgba(0,0,0,0.25);
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: var(--accent-gradient);
        }

        .card-header {
            margin-bottom: 1.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .card-title i {
            color: var(--text-muted);
        }

        /* Inputs Grid */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 640px) {
            .form-grid { grid-template-columns: 1fr; }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .input-wrapper {
            position: relative;
        }

        .input-wrapper i {
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .form-control {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.6rem;
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
        }

        .form-control:focus {
            border-color: var(--border-focus);
            box-shadow: 0 0 15px var(--accent-glow);
        }

        select.form-control {
            appearance: none;
            cursor: pointer;
        }

        /* Premium Interactive Button with Animated Glow & Ripple */
        .btn-wrapper {
            grid-column: span 2;
            margin-top: 1rem;
        }

        @media (max-width: 640px) {
            .btn-wrapper { grid-column: span 1; }
        }

        .btn-premium {
            width: 100%;
            padding: 1.1rem 2rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 16px;
            color: white;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: var(--btn-shadow);
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .btn-premium:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 0 35px var(--accent-glow);
        }

        .btn-premium:active {
            transform: translateY(1px) scale(0.99);
        }

        /* Premium Shine Effect */
        .btn-premium::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                60deg,
                transparent,
                rgba(255, 255, 255, 0.35),
                transparent
            );
            transform: rotate(30deg);
            animation: shine-sweep 3.5s infinite;
        }

        @keyframes shine-sweep {
            0% { transform: translateX(-100%) rotate(30deg); }
            20% { transform: translateX(100%) rotate(30deg); }
            100% { transform: translateX(100%) rotate(30deg); }
        }

        /* Analytics Side Panel */
        .analytics-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .result-card-inner {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 2rem 1.5rem;
            border: 1px solid var(--border-card);
            text-align: center;
            position: relative;
        }

        .result-tag {
            display: inline-block;
            padding: 0.4rem 1.1rem;
            border-radius: 30px;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1.2rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: var(--text-muted);
        }

        .result-status {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .status-churn {
            color: #ef4444;
            text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        }

        .status-retained {
            color: #10b981;
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        }

        .result-subtext {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        /* Feature Weight Bar Visualizer */
        .feature-bars {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            margin-top: 1.5rem;
        }

        .bar-item {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .bar-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .bar-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: var(--accent-gradient);
            border-radius: 10px;
            width: 0%;
            transition: width 1s ease-out;
        }

        /* Spinner Animation */
        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="ambient-glow-1"></div>
    <div class="ambient-glow-2"></div>

    <div class="container">
        <!-- Header -->
        <header>
            <div class="brand-box">
                <div class="brand-icon">
                    <i class="fa-solid fa-brain"></i>
                </div>
                <div class="brand-title">
                    <h1>AI Churn Prediction Studio</h1>
                    <p>Enterprise Machine Learning Analytics</p>
                </div>
            </div>

            <div class="controls-area">
                <i class="fa-solid fa-palette" style="color: var(--text-muted);"></i>
                <select class="theme-select" id="themeSelector" onchange="changeTheme(this.value)">
                    <option value="dark-glass">Dark Glass</option>
                    <option value="emerald">Emerald Luxury</option>
                    <option value="cyberpunk">Cyberpunk Neon</option>
                    <option value="light-minimal">Light Minimal</option>
                </select>
            </div>
        </header>

        <!-- Dashboard Grid -->
        <div class="dashboard-grid">
            
            <!-- Left: Inputs -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-sliders"></i>
                        <span>Customer Profile Attributes</span>
                    </div>
                </div>

                <form id="churnForm">
                    <div class="form-grid">
                        
                        <div class="input-group">
                            <label>Age</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-user"></i>
                                <input type="number" class="form-control" name="Age" value="34" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Gender</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-venus-mars"></i>
                                <select class="form-control" name="Gender">
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Tenure (Months)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-calendar-days"></i>
                                <input type="number" class="form-control" name="Tenure" value="18" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Usage Frequency</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-chart-line"></i>
                                <input type="number" class="form-control" name="Usage Frequency" value="12" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Support Calls</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-headset"></i>
                                <input type="number" class="form-control" name="Support Calls" value="1" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Payment Delay (Days)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-clock-history"></i>
                                <input type="number" class="form-control" name="Payment Delay" value="0" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Subscription Type</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-crown"></i>
                                <select class="form-control" name="Subscription Type">
                                    <option value="Standard">Standard</option>
                                    <option value="Basic">Basic</option>
                                    <option value="Premium">Premium</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Contract Length</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-file-contract"></i>
                                <select class="form-control" name="Contract Length">
                                    <option value="Annual">Annual</option>
                                    <option value="Monthly">Monthly</option>
                                    <option value="Quarterly">Quarterly</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Total Spend ($)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-dollar-sign"></i>
                                <input type="number" step="0.01" class="form-control" name="Total Spend" value="750.00" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Last Interaction (Days)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-hand-pointer"></i>
                                <input type="number" class="form-control" name="Last Interaction" value="3" required>
                            </div>
                        </div>

                        <div class="btn-wrapper">
                            <button type="submit" class="btn-premium" id="submitBtn">
                                <span id="btnText">Analyze Churn Risk</span>
                                <div class="spinner" id="btnSpinner"></div>
                                <i class="fa-solid fa-wand-magic-sparkles"></i>
                            </button>
                        </div>

                    </div>
                </form>
            </div>

            <!-- Right: Results & Dashboard -->
            <div class="analytics-panel">
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fa-solid fa-chart-pie"></i>
                            <span>Prediction Output</span>
                        </div>
                    </div>

                    <div class="result-card-inner">
                        <span class="result-tag" id="resultTag">System Ready</span>
                        <div class="result-status" id="resultDisplay">--</div>
                        <p class="result-subtext" id="subtextDisplay">Run an analysis to generate customer retention insights.</p>
                    </div>

                    <div class="feature-bars">
                        <div class="bar-item">
                            <div class="bar-label">
                                <span>Total Spend Factor</span>
                                <span id="spendVal">0%</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" id="spendBar"></div>
                            </div>
                        </div>

                        <div class="bar-item">
                            <div class="bar-label">
                                <span>Support Friction Score</span>
                                <span id="callsVal">0%</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" id="callsBar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        function changeTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
        }

        document.getElementById('churnForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('btnSpinner');

            btnText.innerText = "Processing...";
            spinner.style.display = "block";
            submitBtn.style.pointerEvents = "none";

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    const display = document.getElementById('resultDisplay');
                    const tag = document.getElementById('resultTag');
                    const subtext = document.getElementById('subtextDisplay');

                    tag.innerText = "Analysis Complete";

                    if (result.prediction === 1) {
                        display.innerText = "High Risk";
                        display.className = "result-status status-churn";
                    } else {
                        display.innerText = "Low Risk";
                        display.className = "result-status status-retained";
                    }

                    if (result.probability !== null) {
                        subtext.innerText = `Prediction Confidence: ${(result.probability * 100).toFixed(1)}%`;
                    }

                    const spend = parseFloat(data["Total Spend"]) || 0;
                    const calls = parseFloat(data["Support Calls"]) || 0;

                    const spendPct = Math.min(100, Math.max(5, (spend / 1000) * 100));
                    const callsPct = Math.min(100, Math.max(5, (calls / 10) * 100));

                    document.getElementById('spendBar').style.width = spendPct + "%";
                    document.getElementById('spendVal').innerText = Math.round(spendPct) + "%";

                    document.getElementById('callsBar').style.width = callsPct + "%";
                    document.getElementById('callsVal').innerText = Math.round(callsPct) + "%";

                } else {
                    alert("Error: " + result.error);
                }
            } catch (err) {
                alert("Server error occurred.");
            } finally {
                btnText.innerText = "Analyze Churn Risk";
                spinner.style.display = "none";
                submitBtn.style.pointerEvents = "auto";
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model missing or failed to load."}), 500

    try:
        data = request.get_json()

        # Map categorical text inputs to original model numbers
        gender_num = CATEGORICAL_MAPPINGS["Gender"].get(data.get("Gender"), 0)
        sub_num = CATEGORICAL_MAPPINGS["Subscription Type"].get(data.get("Subscription Type"), 0)
        contract_num = CATEGORICAL_MAPPINGS["Contract Length"].get(data.get("Contract Length"), 0)

        # Assemble features array in exact sequence expected by model
        features = np.array([[
            float(data.get("Age", 0)),
            gender_num,
            float(data.get("Tenure", 0)),
            float(data.get("Usage Frequency", 0)),
            float(data.get("Support Calls", 0)),
            float(data.get("Payment Delay", 0)),
            sub_num,
            contract_num,
            float(data.get("Total Spend", 0)),
            float(data.get("Last Interaction", 0))
        ]])

        prediction = model.predict(features)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(np.max(model.predict_proba(features)))

        return jsonify({
            "prediction": int(prediction),
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
