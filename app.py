from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model/HDI.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    life_expectancy = float(request.form["life_expectancy"])
    expected_schooling = float(request.form["expected_schooling"])
    mean_schooling = float(request.form["mean_schooling"])
    gni = float(request.form["gni"])

    features = np.array([[life_expectancy,
                          expected_schooling,
                          mean_schooling,
                          gni]])

    prediction = model.predict(features)[0]
    prediction = max(0.0, min(prediction, 1.0))

    if prediction >= 0.800:
        category = "Very High Human Development"
    elif prediction >= 0.700:
        category = "High Human Development"
    elif prediction >= 0.550:
        category = "Medium Human Development"
    else:
        category = "Low Human Development"

    return render_template(
        "index.html",
        prediction_text=f"{prediction:.3f}",
        category=category
    )

if __name__ == "__main__":
    app.run(debug=True)