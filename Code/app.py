from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from adapters.output_adapter import OutputAdapter


# Flask app configuration
app = Flask(
    __name__,
    template_folder="ui/templates",
    static_folder="ui/static"
)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "models", "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ------------------------------
# LAZY LOADERS (fix Render timeout)
# ------------------------------

def load_tabular_model(path):
    import joblib
    return joblib.load(path)


def load_text_model():
    from models.text_model import TextModel
    return TextModel()


def load_image_model(path):
    # load full model saved with torch.save(model)
    model = torch.load(path, map_location="cpu")
    model.eval()
    return model


def load_keras_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path)


def load_metrics():
    from metricss.robustness import (
        robustness_test_tabular, robustness_test_text, robustness_test_image
    )
    from metricss.consistency import (
        consistency_test_tabular, consistency_test_text, consistency_test_image
    )
    from metricss.variance import (
        variance_test_tabular, variance_test_text, variance_test_image
    )
    return {
        "robust_tab": robustness_test_tabular,
        "robust_txt": robustness_test_text,
        "robust_img": robustness_test_image,
        "cons_tab": consistency_test_tabular,
        "cons_txt": consistency_test_text,
        "cons_img": consistency_test_image,
        "var_tab": variance_test_tabular,
        "var_txt": variance_test_text,
        "var_img": variance_test_image,
    }


# ------------------------------
# ROUTES
# ------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_model", methods=["POST"])
def upload_model():
    model_file = request.files["model_file"]
    dataset_file = request.files["dataset_file"]
    model_type = request.form.get("model_type")

    # save uploads
    model_path = os.path.join(app.config["UPLOAD_FOLDER"], model_file.filename)
    model_file.save(model_path)

    dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset_file.filename)
    dataset_file.save(dataset_path)

    # load metrics lazily
    M = load_metrics()
    adapter = OutputAdapter()
    # --------------------------
    # TABULAR
    # --------------------------
    if model_type == "tabular":
        import pandas as pd
        df = pd.read_csv(dataset_path)
        inputs = df.values.tolist()

        model = load_tabular_model(model_path)

        results = {
            "model_type": "Tabular",
            "robustness": M["robust_tab"](model, inputs, adapter),
            "consistency": M["cons_tab"](model, inputs, adapter),
            "variance": M["var_tab"](model, inputs, adapter)
        }


    # --------------------------
    # TEXT
    # --------------------------
    elif model_type == "text":
        with open(dataset_path, "r", encoding="utf-8") as f:
            text_data = f.read()

        model = load_text_model()

        results = {
            "model_type": "Text",
            "robustness": M["robust_txt"](model, text_data, adapter),
            "consistency": M["cons_txt"](model, text_data, adapter),
            "variance": M["var_txt"](model, text_data, adapter)
        }


    # --------------------------
    # IMAGE
    # --------------------------
    elif model_type == "image":
        import zipfile

        # Load full PyTorch model from .pt (torch.save(model))
        model = load_image_model(model_path)

        # unzip images
        extract_folder = dataset_path + "_unzipped"
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(dataset_path, "r") as z:
            z.extractall(extract_folder)

        img_files = [
            f for f in os.listdir(extract_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        img_path = os.path.join(extract_folder, img_files[0])

        # Autodetect input channel requirements
        children = list(model.children())
        if len(children) > 0 and hasattr(children[0], "in_channels"):
            expected_channels = children[0].in_channels
        else:
            expected_channels = 3  # fallback

        # Convert image accordingly
        if expected_channels == 1:
            pil_img = Image.open(img_path).convert("L")
        else:
            pil_img = Image.open(img_path).convert("RGB")

        # use standard 224x224 unless your model needs something else
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        img_tensor = transform(pil_img).unsqueeze(0)  # shape (1,C,H,W)

        results = {
            "model_type": "Image",
            "robustness": M["robust_img"](model, img_tensor, adapter),
            "consistency": M["cons_img"](model, img_tensor, adapter),
            "variance": M["var_img"](model, img_tensor, adapter)
        }


    # --------------------------
    # UNKNOWN TYPE
    # --------------------------
    else:
        results = {"error": "Unknown model type"}

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
