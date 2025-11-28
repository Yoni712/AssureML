from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from PIL import Image

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
    # heavy imports moved inside
    from models.text_model import TextModel
    return TextModel()


def load_image_model():
    # heavy imports moved inside
    from models.image_model import ImageModel
    return ImageModel()


def load_keras_model(path):
    # heavy tensorflow import moved inside
    from tensorflow.keras.models import load_model
    return load_model(path)


# lazy load metric modules
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

    # ROUTING BY MODEL TYPE
    if model_type == "tabular":
        import pandas as pd
        df = pd.read_csv(dataset_path)
        inputs = df.values.tolist()

        model = load_tabular_model(model_path)

        results = {
            "model_type": "Tabular",
            "robustness": M["robust_tab"](model, inputs, None),
            "consistency": M["cons_tab"](model, inputs, None),
            "variance": M["var_tab"](model, inputs, None)
        }

    elif model_type == "text":
        with open(dataset_path, "r", encoding="utf-8") as f:
            text_data = f.read()

        model = load_text_model()

        results = {
            "model_type": "Text",
            "robustness": M["robust_txt"](model, text_data, None),
            "consistency": M["cons_txt"](model, text_data, None),
            "variance": M["var_txt"](model, text_data, None)
        }

    elif model_type == "image":
        import zipfile

        model = load_image_model()

        extract_folder = dataset_path + "_unzipped"
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(dataset_path, "r") as z:
            z.extractall(extract_folder)

        img_files = [f for f in os.listdir(extract_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        img_path = os.path.join(extract_folder, img_files[0])
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img)

        results = {
            "model_type": "Image",
            "robustness": M["robust_img"](model, img_array, None),
            "consistency": M["cons_img"](model, img_array, None),
            "variance": M["var_img"](model, img_array, None)
        }

    else:
        results = {"error": "Unknown model type"}

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
