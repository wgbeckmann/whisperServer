from io import BytesIO

import pytest

from whisper.webserver import create_app


class DummyModel:
    def __init__(self):
        self.last_kwargs = None

    def transcribe(self, path, **kwargs):
        assert path  # ensure path is provided
        self.last_kwargs = kwargs
        return {"text": "dummy"}


@pytest.fixture()
def app():
    app = create_app(model=DummyModel())
    app.config.update(TESTING=True)
    return app


def test_transcribe_endpoint(app):
    client = app.test_client()
    data = {"file": (BytesIO(b"audio"), "test.wav")}
    response = client.post("/transcribe", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json()["text"] == "dummy"


def test_transcribe_with_options(app):
    client = app.test_client()
    data = {"file": (BytesIO(b"audio"), "test.wav")}
    response = client.post(
        "/transcribe?word_timestamps=true&temperature=0.0",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert app.model.last_kwargs["word_timestamps"] is True
    assert app.model.last_kwargs["temperature"] == 0.0


def test_transcribe_model_override():
    calls = {}

    def loader(name):
        calls["name"] = name
        return DummyModel()

    app = create_app(model=DummyModel(), model_loader=loader)
    app.config.update(TESTING=True)
    client = app.test_client()
    data = {"file": (BytesIO(b"audio"), "test.wav")}
    response = client.post(
        "/transcribe?model=small",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert calls["name"] == "small"
