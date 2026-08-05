# Customer-Churn-Predictions
https://customer-churn-predictions-uc0f.onrender.com/

# 📊 AI Churn Analytics Studio

An enterprise-grade, web-based predictive analytics platform built with **Flask**, **Scikit-Learn**, and **Glassmorphism UI**. The application leverages an **AdaBoost Classification model** (`ADABoost.pkl`) to evaluate customer churn risk in real-time based on user engagement metrics and account features.

---

## 🌟 Features

- **Real-Time Machine Learning Scoring**: Instant evaluation of customer churn risk using an AdaBoost ensemble classifier.
- **Glassmorphism UI & Dynamic Theming**: Includes 8 custom, real-time switchable color themes (*Obsidian Cyber, Amethyst Glass, Emerald Executive, Midnight Gold, Rose Gold Luxe, Electric Cyan, Sunset Crimson, Titanium Platinum*).
- **Interactive Feature Radar Profile**: Native HTML5 Canvas radar chart visualizing user feature scales dynamically without external JS library dependencies.
- **Analytical Metrics & Guidance**: Detailed feedback including risk categorizations, probability confidence bars, and automated intervention recommendations.
- **Execution Log Table**: Real-time appending of calculated predictions into a dynamic session history log.
- **Fail-Safe Diagnostic Route**: Integrated `/debug` route to inspect model load status and dependencies during cloud deployment.

---

## 📁 Repository Structure

```text
├── app.py              # Flask server, ML pipeline controller, & frontend interface
├── ADABoost.pkl        # Trained AdaBoost ML Model binary file
├── requirements.txt    # Python package dependencies & pinned versions
└── README.md           # Documentation & project guide
🤖 Input Features & Data MappingThe model accepts 10 feature inputs to predict the classification label (Class 1 vs Class 0):Feature NameDescriptionType / MappingsAgeCustomer age in yearsNumeric (18–100)GenderCustomer genderCategorical (Male: 0, Female: 1)TenureMonths with the platformNumeric (0–120)Usage FrequencyAverage weekly usage sessionsNumeric (0–100)Support CallsTotal customer support inquiriesNumeric (0–50)Payment DelayLate payment history in daysNumeric (0–60)Subscription TypeSubscription plan tierCategorical (Basic: 0, Standard: 1, Premium: 2)Contract LengthBilling contract commitmentCategorical (Monthly: 0, Quarterly: 1, Annual: 2)Total SpendCumulative revenue generated ($)NumericLast InteractionDays since last platform activityNumeric (0–365)⚙️ Requirements & Installation1. PrerequisitesEnsure you have Python installed locally:Python 3.10, 3.11, or 3.12 (Recommended: Python 3.11)2. Dependencies (requirements.txt)PlaintextFlask>=3.0.0
numpy>=1.26.0
scikit-learn==1.6.1
scipy>=1.11.0
gunicorn>=21.2.0
⚠️ Important Note on Scikit-Learn: The model is unpickled using scikit-learn==1.6.1. Ensure the environment uses this exact version to prevent InconsistentVersionWarning or unpickling errors.🚀 Local Development SetupClone the repository:Bashgit clone [https://github.com/YOUR_USERNAME/customer-churn-predictions.git](https://github.com/YOUR_USERNAME/customer-churn-predictions.git)
cd customer-churn-predictions
Create and activate a virtual environment:Bash# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
Install dependencies:Bashpip install -r requirements.txt
Run the Flask development server:Bashpython app.py
Access the application:Open http://localhost:5000 in your web browser.☁️ Deployment Guide (Render)To deploy this web application on Render:Push your code to GitHub: Ensure app.py, ADABoost.pkl, and requirements.txt are in the main root folder.Create a Web Service on Render:Connect your GitHub repository.Environment: Python 3Build Command: pip install -r requirements.txtStart Command: gunicorn app:appConfigure Environment Variables:Go to Settings $\rightarrow$ Environment Variables.Set PYTHON_VERSION to 3.11.0.Deploy:Perform a Clear build cache & deploy to build the environment cleanly.🔍 Diagnostics & DebuggingIf the application ever experiences loading issues on cloud servers, check the built-in diagnostic route:Plaintexthttps://<your-render-domain>/debug
Expected Output: Status: Loaded, Error: NoneIf an error occurs during model loading (e.g., missing file, path mismatch, or version issue), the /debug endpoint will display the exact traceback.
