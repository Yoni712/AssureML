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


def load_text_model(path):
    import joblib
    return joblib.load(path)


def load_image_model(path):
    # PyTorch 2.6 requires disabling weights_only to load full model pickles
    model = torch.load(path, map_location="cpu", weights_only=False)
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

@app.route("/test")
def test_page():
    return render_template("testbench.html")

@app.route("/upload_model", methods=["POST"])
def upload_model():
    model_file = request.files["model_file"]
    dataset_file = request.files["dataset_file"]
    model_type = request.form.get("model_type")
    batch_size = request.form.get("batch_size", "all")

    if batch_size != "all":
        batch_size = int(batch_size)

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
        from adapters.tabular_adapter import TabularAdapter
        tab_adapter = TabularAdapter()

        df = pd.read_csv(dataset_path)

        rows = df.values.tolist()
        if batch_size != "all":
            rows = rows[:batch_size]

        # Convert each row to numpy array shape (1, n)
        processed_rows = [tab_adapter.adapter_input(row) for row in rows]

        model = load_tabular_model(model_path)

        robust_scores = []
        robust_dists = []

        cons_scores = []
        cons_breaks = []

        var_scores = []
        var_lists = []

        # Run metrics per-sample like image
        for sample in processed_rows:
            r_avg, r_dist = M["robust_tab"](model, sample, adapter)
            c_avg, c_break = M["cons_tab"](model, sample, adapter)
            v_avg, v_list = M["var_tab"](model, sample, adapter)

            robust_scores.append(r_avg)
            robust_dists.append(r_dist)

            cons_scores.append(c_avg)
            cons_breaks.append(c_break)

            var_scores.append(v_avg)
            var_lists.append(v_list)

        import numpy as np

        robust_mean = float(np.mean(robust_scores))
        cons_mean = float(np.mean(cons_scores))
        var_mean = float(np.mean(var_scores))

        robust_pct = round(robust_mean * 100, 2)
        cons_pct = round(cons_mean * 100, 2)

        var_norm = max(0, 1 - min(var_mean * 50, 1))
        var_pct = round(var_norm * 100, 2)

        def score_color(p):
            if p >= 85:
                return "good"
            elif p >= 60:
                return "mid"
            else:
                return "bad"

        results = {
            "model_type": "Tabular",
            "batch_evaluated": len(processed_rows),

            "robustness": robust_pct,
            "consistency": cons_pct,
            "variance": var_pct,

            "robustness_color": score_color(robust_pct),
            "consistency_color": score_color(cons_pct),
            "variance_color": score_color(var_pct),

            "robustness_text": "Higher = consistent predictions under feature noise.",
            "consistency_text": "Insensitive to scaling/rounding perturbations.",
            "variance_text": "Low variance = stable model.",

            "raw": {
                "robust_dists": robust_dists,
                "cons_breakdowns": cons_breaks,
                "var_lists": var_lists,
            }
        }




    # --------------------------
    # TEXT
    # --------------------------
    elif model_type == "text":
        from adapters.text_adapter import TextAdapter
        txt_adapter = TextAdapter()

        with open(dataset_path, "r", encoding="utf-8") as f:
            text_data = f.read()

        model = load_text_model(model_path)
        processed_text = txt_adapter.adapter_input(text_data)

        r_avg, r_dist = M["robust_txt"](model, processed_text, adapter)
        c_avg, c_break = M["cons_txt"](model, processed_text, adapter)
        v_avg, v_list = M["var_txt"](model, processed_text, adapter)

        results = {
            "model_type": "Text",
            "batch_evaluated": 1,

            "robustness": round(r_avg * 100, 2),
            "consistency": round(c_avg * 100, 2),
            "variance": round((1 - min(v_avg * 50, 1)) * 100, 2),

            "robustness_color": "good" if r_avg*100 >= 85 else "mid" if r_avg*100 >= 60 else "bad",
            "consistency_color": "good" if c_avg*100 >= 85 else "mid" if c_avg*100 >= 60 else "bad",
            "variance_color": "good" if (1 - min(v_avg * 50, 1)) * 100 >= 85 else "mid" if (1 - min(v_avg * 50, 1)) * 100 >= 60 else "bad",

            "robustness_text": "Shows robustness to typos, word dropout & small text noise.",
            "consistency_text": "Meaning-preserving changes should not flip predictions.",
            "variance_text": "Low variance means stable prediction under micro-typos.",

            "raw": {
                "robust_dists": r_dist,
                "cons_breakdowns": c_break,
                "var_lists": v_list,
            }
        }

    # --------------------------
    # IMAGE
    # --------------------------
    elif model_type == "image":
        import zipfile

        # Load model
        model = load_image_model(model_path)

        # Unzip images
        extract_folder = dataset_path + "_unzipped"
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(dataset_path, "r") as z:
            z.extractall(extract_folder)

        # Collect all images
        img_files = [
            f for f in os.listdir(extract_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Apply batch size
        if batch_size != "all":
            img_files = img_files[:batch_size]

        # Prepare transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Results lists
        robust_scores = []
        robust_dists = []

        cons_scores = []
        cons_breakdowns = []

        var_scores = []
        var_lists = []

        # Run the model on each sample
        for img_name in img_files:
            img_path = os.path.join(extract_folder, img_name)

            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = transform(pil_img).unsqueeze(0)

            # Run each metric (which returns avg + details)
            r_avg, r_dist = M["robust_img"](model, img_tensor, adapter)
            c_avg, c_break = M["cons_img"](model, img_tensor, adapter)
            v_avg, v_list = M["var_img"](model, img_tensor, adapter)

            robust_scores.append(r_avg)
            robust_dists.append(r_dist)

            cons_scores.append(c_avg)
            cons_breakdowns.append(c_break)

            var_scores.append(v_avg)
            var_lists.append(v_list)

        # Aggregate metrics
        import numpy as np

        robust_mean = float(np.mean(robust_scores))
        cons_mean = float(np.mean(cons_scores))
        var_mean = float(np.mean(var_scores))

        # Convert to % with color labels (your existing logic)
        robust_pct = round(robust_mean * 100, 2)
        cons_pct = round(cons_mean * 100, 2)

        var_norm = max(0, 1 - min(var_mean * 50, 1))
        var_pct = round(var_norm * 100, 2)

        def score_color(p):
            if p >= 85:
                return "good"
            elif p >= 60:
                return "mid"
            else:
                return "bad"

        results = {
            "model_type": "Image",
            "batch_evaluated": len(img_files),

            "robustness": robust_pct,
            "robustness_color": score_color(robust_pct),

            "consistency": cons_pct,
            "consistency_color": score_color(cons_pct),

            "variance": var_pct,
            "variance_color": score_color(var_pct),

            "raw": {
                "robust_per_sample": robust_scores,
                "robust_dists": robust_dists,

                "cons_per_sample": cons_scores,
                "cons_breakdowns": cons_breakdowns,

                "var_per_sample": var_scores,
                "var_lists": var_lists,
            }
        }

        return render_template("testbench.html", results=results)



    # --------------------------
    # UNKNOWN TYPE
    # --------------------------
    else:
        results = {"error": "Unknown model type"}

    return render_template("testbench.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
