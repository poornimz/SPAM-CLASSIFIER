from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            # Make prediction
            result = model.predict([message])[0]
            prediction = "Spam" if result == 1 else "Not Spam (Ham)"

    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)