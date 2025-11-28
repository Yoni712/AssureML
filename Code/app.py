from flask import Flask, render_template, request
import os
import joblib
import torch
from PIL import Image
import numpy as np
import torch
from metricss.robustness import robustness_test_tabular, robustness_test_text, robustness_test_image
from metricss.consistency import consistency_test_tabular, consistency_test_text, consistency_test_image
from metricss.variance import variance_test_tabular, variance_test_text, variance_test_image
from tensorflow.keras.models import load_model
from models.simple_cnn import SimpleCNN

class SimpleOutputAdapter:
    def adapt_output(self, y):
        # If PyTorch model returns tensor:
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        y = np.array(y, dtype=float)
        return y.flatten()

# IMPORTANT:
# Tell Flask where to find templates + static inside this same folder
app = Flask(
    __name__,
    template_folder="ui/templates",
    static_folder="ui/static"
)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "models", "uploaded")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_model", methods=["POST"])
def upload_model():
    model_file = request.files["model_file"]
    dataset_file = request.files["dataset_file"]
    model_type = request.form.get("model_type")

    # Save model
    model_path = os.path.join(app.config["UPLOAD_FOLDER"], model_file.filename)
    model_file.save(model_path)

    # Save dataset
    dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset_file.filename)
    dataset_file.save(dataset_path)

    # Load model
    model = load_model(model_path)
    
    # =======================
    #   LOAD DATASET BASED ON TYPE
    # =======================

    if model_type == "tabular":
        import pandas as pd
        df = pd.read_csv(dataset_path)
        inputs = df.values.tolist()

        results = {
            "model_type": "Tabular",
            "robustness": robustness_test_tabular(model, inputs, None),
            "consistency": consistency_test_tabular(model, inputs, None),
            "variance": variance_test_tabular(model, inputs, None)
        }

    elif model_type == "text":
        with open(dataset_path, "r", encoding="utf-8") as f:
            text_data = f.read()

        results = {
            "model_type": "Text",
            "robustness": robustness_test_text(model, text_data, None),
            "consistency": consistency_test_text(model, text_data, None),
            "variance": variance_test_text(model, text_data, None)
        }

    elif model_type == "image":
        import zipfile
        import numpy as np
        from PIL import Image
        
        # Unzip images
        extract_folder = dataset_path + "_unzipped"
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        # Load first image only for testing (simple version)
        img_files = [f for f in os.listdir(extract_folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
        img_path = os.path.join(extract_folder, img_files[0])

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0   # normalize

        # Convert to PyTorch tensor: (H,W,C) → (C,H,W)
        img_tensor = torch.tensor(img_array).permute(2,0,1).unsqueeze(0)

        adapter = SimpleOutputAdapter()

        results = {
            "model_type": "Image",
            "robustness": robustness_test_image(model, img_tensor, adapter),
            "consistency": consistency_test_image(model, img_tensor, adapter),
            "variance": variance_test_image(model, img_tensor, adapter)
        }



    else:
        results = {"error": "Unknown model type"}

    return render_template("index.html", results=results)


def load_model(path):
    """Load model depending on extension"""
    if path.endswith(".pkl") or path.endswith(".joblib"):
        return joblib.load(path)

    if path.endswith(".pt"):
        return torch.load(path, map_location=torch.device("cpu"))

    if path.endswith(".h5"):
        return load_model(path)

    raise ValueError("Unsupported model format: " + path)


if __name__ == "__main__":
    # app.py is inside /ui/, so run it from the project root
    app.run(debug=True)
