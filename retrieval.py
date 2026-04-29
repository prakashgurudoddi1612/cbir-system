"""
Retrieval Engine
================
Loads the pre-computed feature database (fold_3_features.json) and performs
cosine-similarity search against a query feature vector.

JSON schema
-----------
{
  "fold":     <int>,
  "labels":   [int, ...],   # 751 class indices (0-4)
  "features": [[float, ...], ...]   # 751 × 128 feature vectors
}

Because the original JSON does not store file paths we maintain a separate
image-index file (data/image_index.json) that maps list position → absolute
image path.  That index is built automatically when the user provides a
`dataset/` directory (see build_image_index()).
"""

import os
import json
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
FEATURES_PATH = os.path.join(BASE_DIR, "data", "fold_3_features.json")
INDEX_PATH    = os.path.join(BASE_DIR, "data", "image_index.json")
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")

# Human-readable class names (update to match your actual classes)
CLASS_NAMES = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
}

# ── Module-level cache ───────────────────────────────────────────────────────
_features_matrix = None   # (N, D) float32 numpy array
_labels          = None   # (N,)  int list
_image_index     = None   # position → file path  (may be None)


# ── Loader ───────────────────────────────────────────────────────────────────

def _load_database():
    global _features_matrix, _labels, _image_index

    if _features_matrix is not None:
        return

    with open(FEATURES_PATH, "r") as f:
        db = json.load(f)

    _labels          = db["labels"]                                  # list[int]
    _features_matrix = np.array(db["features"], dtype=np.float32)   # (N, D)

    # L2-normalise stored vectors (idempotent if already normalised)
    norms = np.linalg.norm(_features_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    _features_matrix = _features_matrix / norms

    # Try to load the image index
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r") as f:
            idx = json.load(f)
        # Keys are stored as strings in JSON → convert back to int
        _image_index = {int(k): v for k, v in idx.items()}
    else:
        _image_index = {}

    print(f"[Retrieval] Loaded {len(_labels)} feature vectors  "
          f"| dim={_features_matrix.shape[1]}  "
          f"| image_index={'yes' if _image_index else 'no'}")


# ── Index builder ─────────────────────────────────────────────────────────────

def build_image_index(dataset_dir=None, save=True):
    """
    Walk `dataset_dir` (default: ./dataset/) and build a position→path index.

    Expected layout:
        dataset/
          class_0/  img1.jpg  img2.png ...
          class_1/  ...
          ...

    Images are sorted deterministically so the order matches the feature
    extraction order used during training.
    """
    global _image_index, _labels

    _load_database()

    root = dataset_dir or DATASET_DIR
    if not os.path.isdir(root):
        return {"error": f"Dataset directory not found: {root}"}

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Collect images per class, sorted for reproducibility
    class_images = {}
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        imgs = sorted(
            p for p in (os.path.join(entry.path, f)
                        for f in os.listdir(entry.path))
            if os.path.splitext(p)[1].lower() in VALID_EXT
        )
        class_images[entry.name] = imgs

    # Match by class label order (0, 1, 2, …)
    # Build a global ordered list the same way the extractor iterated
    ordered_paths = []
    label_counts  = {}
    for lbl in sorted(set(_labels)):
        dir_name = f"class_{lbl}"
        imgs     = class_images.get(dir_name, [])
        count    = _labels.count(lbl)
        ordered_paths.extend(imgs[:count])
        label_counts[lbl] = min(len(imgs), count)

    # Fill index
    new_index = {i: p for i, p in enumerate(ordered_paths)}

    _image_index = new_index
    if save:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "w") as f:
            json.dump({str(k): v for k, v in new_index.items()}, f)

    return {
        "indexed": len(new_index),
        "label_counts": label_counts,
        "dataset_dir": root,
    }


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(query_vector, top_k=10):
    """
    Given a 1-D float32 query feature vector, return top-k results.

    Returns
    -------
    list of dict:
        {
          "rank":       int,        # 1-based
          "index":      int,        # position in database
          "label":      int,        # class label
          "class_name": str,
          "score":      float,      # cosine similarity ∈ [-1, 1]
          "image_path": str | None, # absolute path or None
          "has_image":  bool,
        }
    """
    _load_database()

    # Normalise query
    qv   = np.array(query_vector, dtype=np.float32)
    norm = np.linalg.norm(qv)
    if norm > 1e-8:
        qv = qv / norm

    # Cosine similarity = dot product (both L2-normalised)
    scores = _features_matrix @ qv          # (N,)

    # Top-K (descending)
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        label      = int(_labels[idx])
        image_path = _image_index.get(int(idx))
        results.append({
            "rank":       rank,
            "index":      int(idx),
            "label":      label,
            "class_name": CLASS_NAMES.get(label, f"Class {label}"),
            "score":      float(scores[idx]),
            "image_path": image_path,
            "has_image":  image_path is not None and os.path.isfile(image_path),
        })

    return results


def database_stats():
    """Return basic statistics about the loaded database."""
    _load_database()
    label_dist = {}
    for lbl in _labels:
        label_dist[CLASS_NAMES.get(lbl, f"Class {lbl}")] = \
            label_dist.get(CLASS_NAMES.get(lbl, f"Class {lbl}"), 0) + 1

    return {
        "total_images":   len(_labels),
        "feature_dim":    int(_features_matrix.shape[1]),
        "num_classes":    len(set(_labels)),
        "class_dist":     label_dist,
        "has_image_index": bool(_image_index),
        "indexed_images": len(_image_index),
    }


def update_class_names(mapping: dict):
    """Dynamically update class name labels, e.g. {0: 'Benign', 1: 'Malignant'}"""
    CLASS_NAMES.update(mapping)
