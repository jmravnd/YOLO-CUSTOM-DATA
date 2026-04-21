from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

results = model("test.jpg")

print(results[0].probs)