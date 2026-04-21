from flask import Flask, render_template, request
import os
import uuid

app = Flask(__name__)

model = None

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO("runs/classify/train/weights/best.pt")
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("static", filename)
    file.save(filepath)

    results = get_model()(filepath)

    probs = results[0].probs
    top_class = probs.top1
    class_name = get_model().names[top_class]
    confidence = float(probs.top1conf) * 100

    return render_template(
        "index.html",
        image=filepath,
        label=class_name,
        confidence=f"{confidence:.2f}%"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
