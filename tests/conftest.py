"""Test setup: makes the tests independent of what is installed or configured.

main_try.py loads TensorFlow, a .env file and real model files at import time. To
keep tests deterministic, this file always:
  - replaces `tensorflow` with a stub (even if TensorFlow is installed), so the
    real model/scaler are never loaded;
  - turns python-dotenv's load_dotenv into a no-op and removes any APIFY_TOKEN_*
    variables, so no .env file or real token is read;
  - blocks outgoing network connections during every test.
The tests then use fake model/scaler objects and a fake ApifyClient.
"""
import os
import socket
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Never read a developer's .env or real tokens.
for _key in [k for k in os.environ if k.startswith("APIFY_TOKEN_")]:
    del os.environ[_key]
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *a, **k: False

# Always use a stub, whether or not TensorFlow is installed.
tf = types.ModuleType("tensorflow")
keras = types.ModuleType("tensorflow.keras")
models = types.ModuleType("tensorflow.keras.models")


def _load_model(*args, **kwargs):
    raise RuntimeError("tensorflow is stubbed in tests")


models.load_model = _load_model
keras.models = models
tf.keras = keras
sys.modules["tensorflow"] = tf
sys.modules["tensorflow.keras"] = keras
sys.modules["tensorflow.keras.models"] = models


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
