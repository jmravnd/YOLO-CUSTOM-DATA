from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load your trained classification model
model = YOLO("runs/classify/train/weights/best.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    results = model(filepath)

    # Get prediction
    probs = results[0].probs
    top_class = probs.top1
    class_name = model.names[top_class]

    # Accuracy / confidence score
    confidence = float(probs.top1conf) * 100  # convert to percentage

    return render_template(
        "index.html",
        image=filepath,
        label=class_name,
        confidence=f"{confidence:.2f}%"
    )

if __name__ == "__main__":
    app.run(debug=True)