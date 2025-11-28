# ASSUREML/api/server.py

from flask import Flask, render_template, request
import os
import joblib
import torch

from metricss.robustness import robustness_test_tabular
from metricss.consistency import consistency_test_tabular
from metricss.variance import variance_test_tabular

app = Flask(__name__, template_folder="../ui/templates", static_folder="../ui/static")

UPLOAD_FOLDER = "../models/uploaded"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_model", methods=["POST"])
def upload_model():
    uploaded = request.files["model_file"]

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded.filename)
    uploaded.save(save_path)

    model = load_model(save_path)

    dummy_input = [[1, 2, 3]]  # replace with real test data later

    results = {
        "robustness": robustness_test_tabular(model, dummy_input, None),
        "consistency": consistency_test_tabular(model, dummy_input, None),
        "variance": variance_test_tabular(model, dummy_input, None),
    }

    return render_template("index.html", results=results)

def load_model(path):
    if path.endswith(".pkl") or path.endswith(".joblib"):
        return joblib.load(path)
    if path.endswith(".pt"):
        return torch.load(path, map_location=torch.device("cpu"))

    raise ValueError("Unsupported model format")

if __name__ == "__main__":
    app.run(debug=True)
