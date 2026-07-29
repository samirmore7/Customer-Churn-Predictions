import os
import pickle
import traceback
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- MODEL INITIALIZATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ADABoost.pkl")

MODEL = None
LOAD_ERROR = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
    else:
        LOAD_ERROR = f"ADABoost.pkl not found at '{MODEL_PATH}'."
except Exception as e:
    LOAD_ERROR = f"Pickle load error: {str(e)}"

# --- MAPPINGS ---
GENDER_MAP = {"Male": 0, "Female": 1}
SUBSCRIPTION_MAP = {"Basic": 0, "Standard": 1, "Premium": 2}
CONTRACT_MAP = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Churn Analytics Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg: #0b0f19;
            --card: rgba(31, 41, 55, 0.6);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #6366f1;
            --text: #f9fafb;
            --muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 1.5rem; min-height: 100vh; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; max-width: 1400px; margin: 0 auto; }
        @media(max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; }
        .form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
        .field { display: flex; flex-direction: column; gap: 0.3rem; }
        label { font-size: 0.72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; }
        input, select { background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 8px; padding: 0.65rem; color: #fff; font-size: 0.875rem; outline: none; }
        input:focus, select:focus { border-color: var(--accent); }
        select option { background: #111827; }
        .btn { grid-column: span 2; background: linear-gradient(135deg, #6366f1, #a855f7); color: #fff; font-weight: 700; padding: 0.85rem; border: none; border-radius: 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
        .result-box { text-align: center; padding: 1.25rem; background: rgba(0,0,0,0.3); border-radius: 12px; border: 1px solid var(--border); margin-bottom: 1rem; }
        .badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
        .pos { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #34d399; }
        .neg { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #f87171; }
        .bar-bg { width: 100%; background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden; margin: 0.75rem 0; }
        .bar-fill { height: 100%; width: 0%; background: var(--accent); transition: width 0.5s ease; }
        .chart-box { height: 220px; display: flex; justify-content: center; align-items: center; }
    </style>
</head>
<body>
    <header>
        <h2><i class="fa-solid fa-chart-pie" style="color:var(--accent)"></i> AI Churn Analytics Studio</h2>
    </header>

    <div class="grid">
        <div class="card">
            <h3 style="margin-bottom:1rem;"><i class="fa-solid fa-sliders"></i> Input Parameters</h3>
            <form id="pForm" onsubmit="runPredict(event)">
                <div class="form-grid">
                    <div class="field"><label>Age</label><input type="number" id="age" value="34" required></div>
                    <div class="field"><label>Gender</label><select id="gender"><option value="Male">Male</option><option value="Female">Female</option></select></div>
                    <div class="field"><label>Tenure (Mos)</label><input type="number" id="tenure" value="24" required></div>
                    <div class="field"><label>Usage Freq</label><input type="number" id="usage" value="18" required></div>
                    <div class="field"><label>Support Calls</label><input type="number" id="calls" value="2" required></div>
                    <div class="field"><label>Payment Delay</label><input type="number" id="delay" value="1" required></div>
                    <div class="field"><label>Subscription</label><select id="sub"><option value="Basic">Basic</option><option value="Standard" selected>Standard</option><option value="Premium">Premium</option></select></div>
                    <div class="field"><label>Contract</label><select id="contract"><option value="Monthly">Monthly</option><option value="Quarterly">Quarterly</option><option value="Annual" selected>Annual</option></select></div>
                    <div class="field"><label>Total Spend ($)</label><input type="number" step="0.01" id="spend" value="850.50" required></div>
                    <div class="field"><label>Last Interaction</label><input type="number" id="inter" value="5" required></div>
                    <button type="submit" class="btn"><i class="fa-solid fa-bolt"></i> Execute Analysis</button>
                </div>
            </form>
        </div>

        <div class="card">
            <h3 style="margin-bottom:1rem;"><i class="fa-solid fa-chart-line"></i> Analytical Output</h3>
            <div class="result-box">
                <div id="badge" class="badge pos"><i class="fa-solid fa-circle-check"></i> <span id="resTitle">Ready for Analysis</span></div>
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:var(--muted);">
                    <span>Confidence Level</span>
                    <span id="confVal">0%</span>
                </div>
                <div class="bar-bg"><div id="bar" class="bar-fill"></div></div>
                <p id="resSub" style="font-size:0.8rem; color:var(--muted);">Click Execute Analysis to generate prediction output.</p>
            </div>
            <div class="chart-box">
                <canvas id="cv" width="280" height="210"></canvas>
            </div>
        </div>
    </div>

    <script>
        function drawChart(vals) {
            const cv = document.getElementById('cv');
            if(!cv) return;
            const ctx = cv.getContext('2d');
            ctx.clearRect(0, 0, 280, 210);
            const cx = 140, cy = 105, r = 70, n = 6;
            const lbls = ['Tenure', 'Usage', 'Calls', 'Delay', 'Spend', 'Interact'];
            
            for(let level = 0.25; level <= 1; level += 0.25) {
                ctx.beginPath();
                for(let i=0; i<n; i++) {
                    let a = (Math.PI*2/n)*i - Math.PI/2;
                    let x = cx + Math.cos(a)*(r*level), y = cy + Math.sin(a)*(r*level);
                    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
                }
                ctx.closePath();
                ctx.strokeStyle = 'rgba(255,255,255,0.1)';
                ctx.stroke();
            }
            
            ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter'; ctx.textAlign = 'center';
            for(let i=0; i<n; i++) {
                let a = (Math.PI*2/n)*i - Math.PI/2;
                ctx.fillText(lbls[i], cx + Math.cos(a)*(r+16), cy + Math.sin(a)*(r+16));
            }

            ctx.beginPath();
            for(let i=0; i<n; i++) {
                let v = Math.min(Math.max(vals[i], 0), 100) / 100;
                let a = (Math.PI*2/n)*i - Math.PI/2;
                let x = cx + Math.cos(a)*(r*v), y = cy + Math.sin(a)*(r*v);
                if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
            }
            ctx.closePath();
            ctx.fillStyle = 'rgba(99, 102, 241, 0.3)'; ctx.fill();
            ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2; ctx.stroke();
        }

        async function runPredict(e) {
            e.preventDefault();
            const payload = {
                Age: parseFloat(document.getElementById('age').value),
                Gender: document.getElementById('gender').value,
                Tenure: parseFloat(document.getElementById('tenure').value),
                Usage_Frequency: parseFloat(document.getElementById('usage').value),
                Support_Calls: parseFloat(document.getElementById('calls').value),
                Payment_Delay: parseFloat(document.getElementById('delay').value),
                Subscription_Type: document.getElementById('sub').value,
                Contract_Length: document.getElementById('contract').value,
                Total_Spend: parseFloat(document.getElementById('spend').value),
                Last_Interaction: parseFloat(document.getElementById('inter').value)
            };

            try {
                const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
                const data = await res.json();

                if(data.status === 'success') {
                    const badge = document.getElementById('badge');
                    const title = document.getElementById('resTitle');
                    const sub = document.getElementById('resSub');
                    const bar = document.getElementById('bar');
                    const conf = document.getElementById('confVal');

                    title.innerText = data.prediction_label;
                    const pct = (data.probability * 100).toFixed(1) + '%';
                    conf.innerText = pct;
                    bar.style.width = pct;

                    if(data.prediction === 1) {
                        badge.className = 'badge pos';
                        sub.innerText = 'High Retention Probability (' + pct + ' confidence)';
                    } else {
                        badge.className = 'badge neg';
                        sub.innerText = 'Elevated Churn Risk (' + pct + ' confidence)';
                    }

                    drawChart([payload.Tenure, payload.Usage_Frequency, payload.Support_Calls*10, payload.Payment_Delay*10, Math.min(payload.Total_Spend/15, 100), payload.Last_Interaction*5]);
                } else {
                    alert('Backend Prediction Error:\n' + data.message);
                }
            } catch(err) {
                alert('Network Error:\n' + err.message);
            }
        }

        window.onload = () => drawRadarChart([24, 18, 2, 1, 85, 5]);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/debug")
def debug():
    return f"Status: {'Loaded' if MODEL else 'Failed'}, Error: {LOAD_ERROR}"

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"status": "error", "message": f"Model uninitialized: {LOAD_ERROR}"}), 500
    try:
        d = request.get_json(force=True)
        feat = np.array([[
            float(d.get("Age", 0)),
            GENDER_MAP.get(d.get("Gender", "Male"), 0),
            float(d.get("Tenure", 0)),
            float(d.get("Usage_Frequency", 0)),
            float(d.get("Support_Calls", 0)),
            float(d.get("Payment_Delay", 0)),
            SUBSCRIPTION_MAP.get(d.get("Subscription_Type", "Basic"), 0),
            CONTRACT_MAP.get(d.get("Contract_Length", "Monthly"), 0),
            float(d.get("Total_Spend", 0)),
            float(d.get("Last_Interaction", 0))
        ]])
        p = int(MODEL.predict(feat)[0])
        prob = float(MODEL.predict_proba(feat)[0][p]) if hasattr(MODEL, "predict_proba") else 1.0
        return jsonify({"status": "success", "prediction": p, "prediction_label": "Class 1" if p == 1 else "Class 0", "probability": prob})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
