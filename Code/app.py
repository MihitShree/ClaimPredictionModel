from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
rf_model_ip = joblib.load('rf_model_ip.pkl')
rf_model_op = joblib.load('rf_model_op.pkl')

@app.route('/predict_ip', methods=['POST'])
def predict_ip():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)  # Debug statement
        df = pd.DataFrame([data])
        print("DataFrame:", df)  # Debug statement
        prediction = rf_model_ip.predict(df)
        print("Prediction:", prediction)  # Debug statement
        return jsonify({'IPAnnualReimbursementAmt': prediction[0]})
    except Exception as e:
        print("Error:", str(e))  # Debug statement
        return jsonify({'error': str(e)}), 400

@app.route('/predict_op', methods=['POST'])
def predict_op():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        prediction = rf_model_op.predict(df)
        return jsonify({'OPAnnualReimbursementAmt': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
