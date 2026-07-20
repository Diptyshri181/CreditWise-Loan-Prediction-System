from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

try:
    model = pickle.load(open("loan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model_columns = list(scaler.feature_names_in_)
except Exception as e:
    print("Error loading files:", e)

@app.route("/")
def home():
    # Jab page khali khulega toh prediction_text None rahega jisse kuck show nahi hoga
    return render_template("index.html", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    income = float(request.form["applicant_income"])
    coapplicant_income = float(request.form["coapplicant_income"])
    savings = float(request.form["savings"])
    credit_score = float(request.form["credit_score"])
    dti_ratio = float(request.form["dti_ratio"])
    
    # Logical check criteria
    if income >= 30000 and credit_score >= 650 and dti_ratio <= 0.40:
        result = "Loan Status: APPROVED. Risk criteria verified successfully."
    else:
        input_data = {col: 0 for col in model_columns}
        
        if "Applicant_Income" in input_data: input_data["Applicant_Income"] = min(income, 15000)
        if "Coapplicant_Income" in input_data: input_data["Coapplicant_Income"] = min(coapplicant_income, 4000)
        if "Savings" in input_data: input_data["Savings"] = min(savings, 12000)
        if "DTI_Ratio_sq" in input_data: input_data["DTI_Ratio_sq"] = dti_ratio ** 2
        if "Credit_Score_sq" in input_data: input_data["Credit_Score_sq"] = credit_score ** 2
        
        df_input = pd.DataFrame([input_data])[model_columns]
        features_scaled = scaler.transform(df_input)
        prediction = model.predict(features_scaled)
        
        if prediction[0] == 1:
            result = "Loan Status: APPROVED. Risk criteria verified successfully."
        else:
            result = "Loan Status: REJECTED. High underwriting risk detected."
        
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, port=8080)