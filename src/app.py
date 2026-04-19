import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from predict import predict_single, load_models

app = Flask(__name__)

# load once at startup
print("Loading models...")
MODELS = load_models()
print("Models ready.")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Risk Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        h3 { margin-top: 0; color: #555; }
        label { display: block; margin: 10px 0 4px; font-size: 14px; color: #666; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; padding: 14px; background: #4CAF50; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 20px; }
        button:hover { background: #45a049; }
        #result { margin-top: 30px; padding: 20px; border-radius: 8px; display: none; }
        .low    { background: #e8f5e9; border: 2px solid #4CAF50; }
        .medium { background: #fff8e1; border: 2px solid #FFC107; }
        .high   { background: #ffebee; border: 2px solid #f44336; }
        .scores { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-top: 15px; }
        .score-box { background: white; padding: 12px; border-radius: 6px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .score-box .val { font-size: 22px; font-weight: bold; color: #333; }
        .score-box .lbl { font-size: 12px; color: #888; margin-top: 4px; }
        .progress-bar { background: #e0e0e0; border-radius: 10px; height: 20px; margin-top: 15px; }
        .progress-fill { height: 100%; border-radius: 10px; transition: width 0.5s; }
        #loading { display: none; text-align: center; margin-top: 20px; color: #666; }
    </style>
</head>
<body>
    <h1>💳 Credit Risk Predictor</h1>
    <p style="color:#888">Home Credit Default Risk — Stacked Ensemble (LGB + XGB + CatBoost) | OOF AUC: 0.7881</p>

    <form id="predictForm">
    <div class="form-grid">
        <div class="section">
            <h3>Personal Info</h3>
            <label>Gender</label>
            <select name="gender"><option value="M">Male</option><option value="F">Female</option></select>
            <label>Age</label>
            <input type="number" name="age" value="35" min="18" max="70">
            <label>Education</label>
            <select name="education">
                <option>Higher education</option>
                <option>Secondary / secondary special</option>
                <option>Incomplete higher</option>
                <option>Lower secondary</option>
                <option>Academic degree</option>
            </select>
            <label>Family Status</label>
            <select name="family_status">
                <option>Married</option>
                <option>Single / not married</option>
                <option>Civil marriage</option>
                <option>Separated</option>
                <option>Widow</option>
            </select>
            <label>Family Members</label>
            <input type="number" name="family_members" value="2" min="1" max="10">
        </div>

        <div class="section">
            <h3>Employment & Assets</h3>
            <label>Years Employed</label>
            <input type="number" name="employment_years" value="5" min="0" max="40">
            <label>Annual Income</label>
            <input type="number" name="income" value="300000" min="1">
            <label>Owns Car</label>
            <select name="owns_car"><option value="Y">Yes</option><option value="N">No</option></select>
            <label>Owns House/Realty</label>
            <select name="owns_house"><option value="Y">Yes</option><option value="N">No</option></select>
        </div>

        <div class="section">
            <h3>Loan Details</h3>
            <label>Loan Amount</label>
            <input type="number" name="loan_amount" value="500000" min="1">
            <label>Monthly Annuity</label>
            <input type="number" name="annuity" value="25000" min="1">
            <label>External Score 1 (0-1)</label>
            <input type="number" name="ext_source_1" value="0.5" min="0" max="1" step="0.01">
            <label>External Score 2 (0-1)</label>
            <input type="number" name="ext_source_2" value="0.5" min="0" max="1" step="0.01">
            <label>External Score 3 (0-1)</label>
            <input type="number" name="ext_source_3" value="0.5" min="0" max="1" step="0.01">
        </div>
    </div>

    <button type="submit">🔍 Predict Default Risk</button>
    </form>

    <div id="loading">⏳ Running prediction...</div>

    <div id="result">
        <h2 id="result-title"></h2>
        <div class="scores">
            <div class="score-box"><div class="val" id="stacked-score"></div><div class="lbl">Stacked Score</div></div>
            <div class="score-box"><div class="val" id="lgb-score"></div><div class="lbl">LightGBM</div></div>
            <div class="score-box"><div class="val" id="xgb-score"></div><div class="lbl">XGBoost</div></div>
            <div class="score-box"><div class="val" id="cat-score"></div><div class="lbl">CatBoost</div></div>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <p id="decision" style="font-size:18px; font-weight:bold; margin-top:15px;"></p>
    </div>

    <script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        document.getElementById('loading').style.display = 'block';
        document.getElementById('result').style.display = 'none';

        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());

        const resp = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const r = await resp.json();

        document.getElementById('loading').style.display = 'none';

        if (r.error) {
            alert('Error: ' + r.error);
            return;
        }

        const resultDiv = document.getElementById('result');
        resultDiv.className = r.risk_label === 'Low Risk' ? 'low' : r.risk_label === 'Medium Risk' ? 'medium' : 'high';
        resultDiv.style.display = 'block';

        const emoji = r.risk_label === 'Low Risk' ? '✅' : r.risk_label === 'Medium Risk' ? '⚠️' : '🚨';
        document.getElementById('result-title').textContent = emoji + ' ' + r.risk_label;
        document.getElementById('stacked-score').textContent = r.stacked.toFixed(4);
        document.getElementById('lgb-score').textContent = r.lgb.toFixed(4);
        document.getElementById('xgb-score').textContent = r.xgb.toFixed(4);
        document.getElementById('cat-score').textContent = r.cat.toFixed(4);

        const pct = Math.min(r.stacked * 100, 100);
        const color = r.risk_label === 'Low Risk' ? '#4CAF50' : r.risk_label === 'Medium Risk' ? '#FFC107' : '#f44336';
        document.getElementById('progress-fill').style.width = pct + '%';
        document.getElementById('progress-fill').style.background = color;

        document.getElementById('decision').textContent = r.is_default ? '❌ Decision: REJECT' : '✅ Decision: APPROVE';
    });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        app_df = pd.DataFrame({
            "SK_ID_CURR":          [999999],
            "CODE_GENDER":         [data["gender"]],
            "DAYS_BIRTH":          [-int(data["age"]) * 365],
            "DAYS_EMPLOYED":       [-int(data["employment_years"]) * 365 if int(data["employment_years"]) > 0 else 365243],
            "AMT_INCOME_TOTAL":    [float(data["income"])],
            "AMT_CREDIT":          [float(data["loan_amount"])],
            "AMT_ANNUITY":         [float(data["annuity"])],
            "CNT_FAM_MEMBERS":     [float(data["family_members"])],
            "FLAG_OWN_CAR":        [data["owns_car"]],
            "FLAG_OWN_REALTY":     [data["owns_house"]],
            "NAME_EDUCATION_TYPE": [data["education"]],
            "NAME_FAMILY_STATUS":  [data["family_status"]],
            "OWN_CAR_AGE":         [None],
            "EXT_SOURCE_1":        [float(data["ext_source_1"])],
            "EXT_SOURCE_2":        [float(data["ext_source_2"])],
            "EXT_SOURCE_3":        [float(data["ext_source_3"])],
        })

        result = predict_single(app_df, models=MODELS)
        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=False, port=5001)