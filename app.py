import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", device="Render Web")

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

    return jsonify({
        "results": [],
        "preference_count": len(preference_files),
        "resource_count": len(resource_files),
        "device": "Render Web",
        "note": "PinToStay 웹 서버가 정상적으로 작동하고 있습니다."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
