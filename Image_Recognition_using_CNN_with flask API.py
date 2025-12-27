from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os

app = Flask(__name__)
model = ResNet50(weights="imagenet")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    label = confidence = None

    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = image.load_img(path, target_size=(224,224))
        x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

        preds = model.predict(x)
        label, confidence = decode_predictions(preds, 1)[0][0][1:]
        confidence = round(confidence * 100, 2) 

    return render_template("index.html", label=label, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
