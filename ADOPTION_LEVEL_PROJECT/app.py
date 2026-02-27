
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("adoption_model.pkl")

# Label mapping (same as notebook)
label_map = {0: "High", 1: "Medium", 2: "Low"}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # EXACT SAME ORDER as X_train_smote.columns
        features = [
            float(request.form["login_frequency"]),
            float(request.form["session_duration_min"]),
            float(request.form["feature_usage_count"]),
            float(request.form["tasks_completed"]),
            float(request.form["Subscription_Type"]),
            float(request.form["days_since_Signup"]),
            float(request.form["support_tickets"]),
            float(request.form["training_attended"])
        ]

        final_input = np.array([features])

        prediction = model.predict(final_input)
        predicted_label = label_map[prediction[0]]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Adoption Level: {predicted_label}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error occurred: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)