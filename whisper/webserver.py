import os
import tempfile

from flask import Flask, jsonify, request

import whisper


def _parse_param(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def create_app(model=None, model_loader=None):
    app = Flask(__name__)
    loader = model_loader or whisper.load_model
    app.model = model or loader(os.getenv("WHISPER_MODEL", "base"))
    app.model_loader = loader

    @app.route("/transcribe", methods=["POST"])
    def transcribe_route():
        if "file" not in request.files:
            return jsonify({"error": "file field required"}), 400

        file = request.files["file"]
        suffix = os.path.splitext(file.filename)[1]

        # query and form params for transcribe options
        params = {**request.args.to_dict(flat=True), **request.form.to_dict(flat=True)}
        model_name = params.pop("model", None)
        options = {k: _parse_param(v) for k, v in params.items()}

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_name = tmp.name
        try:
            model_inst = app.model_loader(model_name) if model_name else app.model
            result = model_inst.transcribe(temp_name, **options)
        finally:
            os.remove(temp_name)

        return jsonify({"text": result.get("text")})

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
