from io import BytesIO

import pytest

from whisper.webserver import create_app


class DummyModel:
    def transcribe(self, path):
        assert path  # ensure path is provided
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
