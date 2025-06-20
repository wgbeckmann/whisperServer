import inspect
import os
import tempfile

from flask import Flask, jsonify, request

import whisper
from whisper.decoding import DecodingOptions


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


def available_parameters():
    sig = inspect.signature(whisper.transcribe)
    params = [
        name
        for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    decode_params = list(DecodingOptions.__dataclass_fields__.keys())
    return sorted({"model", "timestamps", *params, *decode_params})


def print_available_parameters():
    joined = ", ".join(available_parameters())
    print(f"Available parameters for /transcribe: {joined}")


def create_app(model=None, model_loader=None):
    check_cuda()
    app = Flask(__name__)
    loader = model_loader or whisper.load_model
    app.model = model or loader(os.getenv("WHISPER_MODEL", "turbo"))
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
        timestamps = _parse_param(params.pop("timestamps", "false"))
        options = {k: _parse_param(v) for k, v in params.items()}

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_name = tmp.name
        try:
            model_inst = app.model_loader(model_name) if model_name else app.model
            result = model_inst.transcribe(temp_name, **options)
        finally:
            os.remove(temp_name)
           
        print("Returning result:", result.get("text", "No text found"))
        if timestamps:

            def _format(t: float) -> str:
                secs = int(t)
                hours = secs // 3600
                minutes = (secs % 3600) // 60
                seconds = secs % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            lines = [
                f"{_format(seg['start'])}: {seg['text'].strip()}"
                for seg in result.get("segments", [])
            ]
            text = "\n".join(lines)
        else:
            text = result.get("text")

        return jsonify({"text": text})

    return app


if __name__ == "__main__":
    print_available_parameters()
    create_app().run(host="0.0.0.0", port=8001)
