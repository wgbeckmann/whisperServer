import os
import tempfile

from flask import Flask, jsonify, request

import whisper

def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA is available.")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available.")
    except ImportError:
        print("PyTorch is not installed. Cannot check CUDA availability.")

def create_app(model=None):
    app = Flask(__name__)
    app.model = model or whisper.load_model(os.getenv("WHISPER_MODEL", "turbo"))
    check_cuda()

    @app.route("/transcribe", methods=["POST"])
    def transcribe_route():
        if "file" not in request.files:
            return jsonify({"error": "file field required"}), 400

        file = request.files["file"]
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_name = tmp.name
        try:
            print("Transcribing...")
            result = app.model.transcribe(temp_name)
            print("Transcription complete.")
        finally:
            os.remove(temp_name)
           
        print("Returning result:", result.get("text", "No text found"))
        return jsonify({"text": result.get("text")})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8001)
