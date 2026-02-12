from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()

@app.route("/")
def home():
    return render_template("index.html", model_accuracy=accuracy)


@app.route("/predict", methods=["POST"])
def predict():

    message = ""

    # Case 1: If user typed message
    if "message" in request.form and request.form["message"].strip() != "":
        message = request.form["message"]

    # Case 2: If user uploaded file
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            message = file.read().decode("utf-8")

    # If nothing provided
    if message == "":
        return render_template(
            "index.html",
            prediction_text="Please enter message or upload file.",
            model_accuracy=accuracy
        )

    # Vectorize
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    # Get probability scores
    probability = model.predict_proba(message_vector)
    spam_prob = probability[0][1]
    ham_prob = probability[0][0]

    if prediction[0] == 1:
        result = "Spam Message ❌"
    else:
        result = "Not Spam ✅"
    
    result_text = f"""
    {result} <br>
    Spam Probability: {spam_prob:.2f} <br>
    Ham Probability: {ham_prob:.2f}
    """

    return render_template(
        "index.html",
        prediction_text=result,
        model_accuracy=accuracy
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

