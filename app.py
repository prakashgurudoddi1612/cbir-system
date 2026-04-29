"""
CBIR Flask Application
======================
Content-Based Image Retrieval system powered by DenseNet121 features.
"""

import os
import uuid
import base64
import json
from io import BytesIO
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, url_for, abort
)
from werkzeug.utils import secure_filename
from PIL import Image

from feature_extractor import extract_features, model_available
from retrieval import retrieve, database_stats, build_image_index, update_class_names

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
DATASET_DIR  = BASE_DIR / "dataset"

UPLOAD_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

ALLOWED_EXT  = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
MAX_FILE_MB  = 16

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024
app.config["SECRET_KEY"]         = os.urandom(24)


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def image_to_base64(path: str, size=(260, 260)) -> str | None:
    """Resize and encode an image to base64 data URI for inline display."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def pil_to_base64(img: Image.Image, size=(420, 420)) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = database_stats()
    return render_template("index.html", stats=stats, model_ok=model_available())


@app.route("/retrieve", methods=["POST"])
def retrieve_images():
    if "query_image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["query_image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 415

    # ── Save query image ──
    ext       = file.filename.rsplit(".", 1)[1].lower()
    uid       = uuid.uuid4().hex[:12]
    filename  = f"query_{uid}.{ext}"
    save_path = UPLOAD_DIR / filename
    file.save(str(save_path))

    try:
        # ── Extract features ──
        query_vec = extract_features(str(save_path))

        # ── Similarity search ──
        top_k   = int(request.form.get("top_k", 10))
        results = retrieve(query_vec, top_k=top_k)

        # ── Encode query image ──
        query_img    = Image.open(str(save_path))
        query_b64    = pil_to_base64(query_img)
        query_size   = query_img.size

        # ── Encode result images (if available on disk) ──
        for r in results:
            if r["has_image"]:
                r["image_data"] = image_to_base64(r["image_path"])
            else:
                r["image_data"] = None
            # Remove raw path from client response for security
            del r["image_path"]

        return jsonify({
            "query_image":  query_b64,
            "query_size":   query_size,
            "query_file":   filename,
            "results":      results,
            "total_db":     database_stats()["total_images"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    return jsonify(database_stats())


@app.route("/configure", methods=["POST"])
def configure():
    """Update class names and optionally re-index dataset."""
    data = request.get_json(silent=True) or {}

    if "class_names" in data:
        mapping = {int(k): v for k, v in data["class_names"].items()}
        update_class_names(mapping)

    if data.get("rebuild_index"):
        result = build_image_index(data.get("dataset_dir"))
        return jsonify({"ok": True, "index_result": result})

    return jsonify({"ok": True})


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  CBIR System — Content-Based Image Retrieval")
    print("═" * 60)
    stats = database_stats()
    print(f"  Database  : {stats['total_images']} images  |  {stats['feature_dim']}-dim features")
    print(f"  Classes   : {stats['num_classes']}")
    print(f"  Model     : {'✓ loaded' if model_available() else '⚠ not found (demo mode)'}")
    print(f"  URL       : http://127.0.0.1:5000")
    print("═" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
