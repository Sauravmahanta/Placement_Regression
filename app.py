# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("Placement_Regression.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Serve HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:  # API request
            data = request.get_json()
            cgpa = data.get("cgpa")
        else:  # Form submission
            cgpa = float(request.form.get("cgpa"))

        if cgpa is None:
            return jsonify({"error": "CGPA is required"}), 400

        # Prediction
        cgpa_array = np.array([[cgpa]])
        predicted_salary = model.predict(cgpa_array)[0]

        if request.is_json:  # JSON API response
            return jsonify({
                "cgpa": cgpa,
                "predicted_salary": round(predicted_salary, 2)
            })
        else:  # Render result on webpage
            return render_template("index.html", cgpa=cgpa, salary=round(predicted_salary, 2))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
