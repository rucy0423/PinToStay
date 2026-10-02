import os
from io import BytesIO

import numpy as np
import torch
import open_clip
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB total upload limit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, _, PREPROCESS = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
MODEL = MODEL.to(DEVICE)
MODEL.eval()


def image_embedding(image):
    """Create a normalized CLIP image embedding from a PIL image."""
    image = image.convert("RGB")
    image_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = MODEL.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return embedding.cpu().numpy()[0]


def load_upload(file_storage):
    raw = file_storage.read()
    image = Image.open(BytesIO(raw))
    image.verify()
    image = Image.open(BytesIO(raw)).convert("RGB")
    return image


@app.route("/")
def index():
    return render_template("index.html", device=DEVICE)


@app.route("/analyze", methods=["POST"])
def analyze():
    preference_files = request.files.getlist("preferences")
    resource_files = request.files.getlist("resources")

    preference_files = [f for f in preference_files if f and f.filename]
    resource_files = [f for f in resource_files if f and f.filename]

    if not preference_files:
        return jsonify({"error": "선호 사진을 최소 1장 업로드해 주세요."}), 400
    if not resource_files:
        return jsonify({"error": "비교할 관광자원 사진을 최소 1장 업로드해 주세요."}), 400

    try:
        preference_vectors = [image_embedding(load_upload(f)) for f in preference_files]
        user_vector = np.mean(preference_vectors, axis=0)
        norm = np.linalg.norm(user_vector)
        if norm < 1e-12:
            return jsonify({"error": "선호 사진을 분석하지 못했어요. 다른 이미지를 사용해 주세요."}), 400
        user_vector = user_vector / norm

        results = []
        for file in resource_files:
            image = load_upload(file)
            poi_vector = image_embedding(image)
            score = float(np.dot(user_vector, poi_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(poi_vector) + 1e-12
            ))
            results.append({
                "name": os.path.basename(file.filename),
                "score": round(score, 4),
                "score_percent": max(0, min(100, round(score * 100, 1))),
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        for i, result in enumerate(results, start=1):
            result["rank"] = i

        return jsonify({
            "results": results,
            "preference_count": len(preference_files),
            "resource_count": len(resource_files),
            "device": DEVICE,
            "note": "유사도는 CLIP 이미지 임베딩 간 코사인 유사도입니다. 백분율은 취향 일치율이나 추천 정확도가 아닙니다."
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "이미지를 읽지 못했어요. JPG, PNG, WEBP 등 일반 이미지 파일을 사용해 주세요."}), 400
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify({"error": "분석 중 오류가 발생했어요. 서버 터미널의 오류 메시지를 확인해 주세요."}), 500


if __name__ == "__main__":
    # Local development server. Keep debug=False for safer default behavior.
    app.run(host="127.0.0.1", port=5000, debug=False)
