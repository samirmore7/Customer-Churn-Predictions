import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ADABoost.pkl")
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Mappings for categorical features to numerical values expected by AdaBoost
CATEGORICAL_MAPPINGS = {
    "Gender": {"Female": 0, "Male": 1},
    "Subscription Type": {"Basic": 0, "Standard": 1, "Premium": 2},
    "Contract Length": {"Monthly": 0, "Quarterly": 1, "Annual": 2}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost AI Predictive Engine</title>
    <!-- Google Fonts & FontAwesome Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            /* Dark Theme Colors (Default) */
            --bg-base: #0a0d14;
            --bg-surface: rgba(18, 24, 38, 0.7);
            --bg-card: rgba(26, 34, 53, 0.6);
            --border-glow: rgba(99, 102, 241, 0.25);
            --border-card: rgba(255, 255, 255, 0.08);
            
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-glow: #a5b4fc;

            --accent-primary: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --accent-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 50%, #db2777 100%);
            
            --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --shadow-neon: 0 0 20px rgba(99, 102, 241, 0.4);
        }

        [data-theme="light"] {
            /* Light Theme Colors */
            --bg-base: #f0f4f9;
            --bg-surface: rgba(255, 255, 255, 0.85);
            --bg-card: rgba(255, 255, 255, 0.65);
            --border-glow: rgba(99, 102, 241, 0.2);
            --border-card: rgba(0, 0, 0, 0.08);
            
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --text-glow: #4f46e5;

            --shadow-glass: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
            --shadow-neon: 0 4px 20px rgba(99, 102, 241, 0.25);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.12) 0%, transparent 40%);
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Navbar */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2rem;
            background: var(--bg-surface);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid var(--border-card);
            box-shadow: var(--shadow-glass);
            margin-bottom: 2.5rem;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: var(--accent-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
            box-shadow: var(--shadow-neon);
        }

        .logo-text h1 {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-text p {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .theme-toggle-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-main);
            padding: 0.6rem 1.2rem;
            border-radius: 30px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .theme-toggle-btn:hover {
            border-color: var(--accent-primary);
            box-shadow: var(--shadow-neon);
            transform: translateY(-2px);
        }

        /* Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 420px;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glass Card Base */
        .glass-card {
            background: var(--bg-surface);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            border: 1px solid var(--border-card);
            padding: 2rem;
            box-shadow: var(--shadow-glass);
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
            color: var(--accent-primary);
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
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
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        select.form-control {
            appearance: none;
            cursor: pointer;
        }

        /* Animated Premium Button */
        .btn-submit {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem 2rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 14px;
            color: white;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: var(--shadow-neon);
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @media (max-width: 640px) {
            .btn-submit {
                grid-column: span 1;
            }
        }

        .btn-submit:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .btn-submit::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                60deg,
                transparent,
                rgba(255, 255, 255, 0.25),
                transparent
            );
            transform: rotate(30deg);
            animation: shine 4s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%) rotate(30deg); }
            20% { transform: translateX(100%) rotate(30deg); }
            100% { transform: translateX(100%) rotate(30deg); }
        }

        /* Right Panel / Output Area */
        .results-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .result-box {
            background: var(--bg-card);
            border-radius: 18px;
            padding: 1.75rem;
            border: 1px solid var(--border-card);
            text-align: center;
            position: relative;
        }

        .result-badge {
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
            background: rgba(99, 102, 241, 0.15);
            color: var(--text-glow);
            border: 1px solid var(--border-glow);
        }

        .result-value {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .result-subtext {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Pulse Animations for Predictions */
        .status-positive {
            color: #10b981;
        }
        .status-negative {
            color: #f43f5e;
        }

        /* Feature Weight Mini Dashboard */
        .metrics-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 1rem;
        }

        .metric-item {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }

        .metric-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .metric-bar-bg {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .metric-bar-fill {
            height: 100%;
            background: var(--accent-gradient);
            border-radius: 4px;
            width: 0%;
            transition: width 1s ease-in-out;
        }

        /* Loader */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="container">
        <!-- Header -->
        <header>
            <div class="logo-area">
                <div class="logo-icon">
                    <i class="fa-solid fa-bolt"></i>
                </div>
                <div class="logo-text">
                    <h1>AdaBoost Intelligence</h1>
                    <p>Ensemble Model Prediction Dashboard</p>
                </div>
            </div>

            <button class="theme-toggle-btn" id="themeToggleBtn" onclick="toggleTheme()">
                <i class="fa-solid fa-moon"></i>
                <span id="themeText">Dark Mode</span>
            </button>
        </header>

        <!-- Main Body Grid -->
        <div class="dashboard-grid">
            
            <!-- Left Panel: Input Controls -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-sliders"></i>
                        <span>Model Parameters</span>
                    </div>
                </div>

                <form id="predictionForm">
                    <div class="form-grid">
                        
                        <div class="input-group">
                            <label>Age</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-user"></i>
                                <input type="number" class="form-control" name="Age" value="30" required>
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
                                <input type="number" class="form-control" name="Tenure" value="12" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Usage Frequency</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-chart-line"></i>
                                <input type="number" class="form-control" name="Usage Frequency" value="15" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Support Calls</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-headset"></i>
                                <input type="number" class="form-control" name="Support Calls" value="2" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Payment Delay (Days)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-clock-history"></i>
                                <input type="number" class="form-control" name="Payment Delay" value="1" required>
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
                                <input type="number" step="0.01" class="form-control" name="Total Spend" value="500.00" required>
                            </div>
                        </div>

                        <div class="input-group">
                            <label>Last Interaction (Days)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-hand-pointer"></i>
                                <input type="number" class="form-control" name="Last Interaction" value="5" required>
                            </div>
                        </div>

                        <button type="submit" class="btn-submit" id="submitBtn">
                            <span id="btnText">Execute Prediction</span>
                            <div class="spinner" id="btnSpinner"></div>
                            <i class="fa-solid fa-wand-magic-sparkles"></i>
                        </button>

                    </div>
                </form>
            </div>

            <!-- Right Panel: Results & Analytics -->
            <div class="results-panel">
                
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fa-solid fa-square-poll-vertical"></i>
                            <span>Prediction Insights</span>
                        </div>
                    </div>

                    <div class="result-box">
                        <span class="result-badge" id="statusBadge">Awaiting Input</span>
                        <div class="result-value" id="resultDisplay">--</div>
                        <p class="result-subtext" id="confidenceDisplay">Fill out model parameters and run inference.</p>
                    </div>

                    <div class="metrics-list" id="metricsContainer">
                        <div class="metric-item">
                            <div class="metric-label">
                                <span>Total Spend Contribution</span>
                                <span id="spendVal">50%</span>
                            </div>
                            <div class="metric-bar-bg">
                                <div class="metric-bar-fill" id="spendBar" style="width: 50%;"></div>
                            </div>
                        </div>

                        <div class="metric-item">
                            <div class="metric-label">
                                <span>Support Call Friction</span>
                                <span id="callsVal">20%</span>
                            </div>
                            <div class="metric-bar-bg">
                                <div class="metric-bar-fill" id="callsBar" style="width: 20%;"></div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

        </div>
    </div>

    <script>
        // Theme Switcher Logic
        function toggleTheme() {
            const html = document.documentElement;
            const themeBtn = document.getElementById('themeToggleBtn');
            const themeText = document.getElementById('themeText');
            
            if (html.getAttribute('data-theme') === 'dark') {
                html.setAttribute('data-theme', 'light');
                themeBtn.innerHTML = '<i class="fa-solid fa-sun"></i> <span>Light Mode</span>';
            } else {
                html.setAttribute('data-theme', 'dark');
                themeBtn.innerHTML = '<i class="fa-solid fa-moon"></i> <span>Dark Mode</span>';
            }
        }

        // Form Submit API Request
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('btnSpinner');

            // Set Loading UI state
            btnText.innerText = "Analyzing...";
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
                    const badge = document.getElementById('statusBadge');
                    const confidence = document.getElementById('confidenceDisplay');

                    badge.innerText = "Inference Complete";
                    
                    if (result.prediction === 1) {
                        display.innerText = "Class 1 Detected";
                        display.className = "result-value status-positive";
                    } else {
                        display.innerText = "Class 0 Detected";
                        display.className = "result-value status-negative";
                    }

                    if (result.probability !== null) {
                        confidence.innerText = `Confidence Score: ${(result.probability * 100).toFixed(1)}%`;
                    } else {
                        confidence.innerText = "Prediction calculated successfully.";
                    }

                    // Dynamically animate analytics metrics based on input
                    const spend = parseFloat(data["Total Spend"]) || 0;
                    const calls = parseFloat(data["Support Calls"]) || 0;
                    
                    const spendPct = Math.min(100, Math.max(10, (spend / 1000) * 100));
                    const callsPct = Math.min(100, Math.max(10, (calls / 10) * 100));

                    document.getElementById('spendBar').style.width = spendPct + "%";
                    document.getElementById('spendVal').innerText = Math.round(spendPct) + "%";
                    
                    document.getElementById('callsBar').style.width = callsPct + "%";
                    document.getElementById('callsVal').innerText = Math.round(callsPct) + "%";

                } else {
                    alert("Prediction Error: " + result.error);
                }
            } catch (err) {
                alert("Server Connection Failed.");
            } finally {
                btnText.innerText = "Execute Prediction";
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
        return jsonify({"error": "Model file not found or failed to load."}), 500

    try:
        data = request.get_json()

        # Parse categorical features back to model numerical values
        gender_num = CATEGORICAL_MAPPINGS["Gender"].get(data.get("Gender"), 0)
        sub_num = CATEGORICAL_MAPPINGS["Subscription Type"].get(data.get("Subscription Type"), 0)
        contract_num = CATEGORICAL_MAPPINGS["Contract Length"].get(data.get("Contract Length"), 0)

        # Assemble features array in exact model order
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
