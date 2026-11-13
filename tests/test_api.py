"""Mocked tests for the InsPector API (main_try.py).

These check request handling, feature extraction, error paths and the decision
bands with fake data. They do NOT prove the real model is accurate or that the
live Apify scrapers behave the same way.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main_try


class FakeScaler:
    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = np.array(values, dtype=float)
        return self.seen


class FakeModel:
    """Marks the first `bots` rows as bots (score 0.9), the rest as real (0.1)."""

    def __init__(self, bots=0):
        self.bots = bots

    def predict(self, data):
        scores = [[0.9] if i < self.bots else [0.1] for i in range(len(data))]
        return np.array(scores)


class FakeApify:
    """Replaces ApifyClient. `actors` maps actor id -> list of dataset items."""

    actors = {}

    def __init__(self, token):
        self.token = token

    def actor(self, actor_id):
        outer = self

        class _Actor:
            def call(self, run_input=None):
                return {"defaultDatasetId": actor_id}

        return _Actor()

    def dataset(self, dataset_id):
        items = FakeApify.actors.get(dataset_id, [])

        class _Dataset:
            def iterate_items(self):
                return iter(items)

        return _Dataset()


FOLLOWERS = "scraping_solutions/instagram-scraper-followers-following-no-cookies"
PROFILE = "apify/instagram-profile-scraper"


def make_profiles(n):
    return [
        {
            "username": f"user{i}",
            "fullName": f"User {i}",
            "biography": "hello",
            "isPrivate": False,
            "profilePicUrl": "https://example.com/pic.jpg",
            "externalUrl": None,
            "postsCount": 3,
            "followersCount": 10,
            "followsCount": 20,
        }
        for i in range(n)
    ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_try, "model", FakeModel(bots=0))
    monkeypatch.setattr(main_try, "scaler", FakeScaler())
    monkeypatch.setattr(main_try, "ASSET_ERROR", None)
    monkeypatch.setattr(main_try, "DAFTAR_TOKEN", ["fake-token"])
    monkeypatch.setattr(main_try, "ApifyClient", FakeApify)
    FakeApify.actors = {}
    return TestClient(main_try.app)


def setup_account(n=50, private=False):
    target = {"username": "target", "isPrivate": private}
    profiles = make_profiles(n)
    FakeApify.actors = {
        FOLLOWERS: [{"username": p["username"]} for p in profiles],
    }
    # Both profile-scraper calls use the same actor id: the first call checks the
    # target account, the second fetches follower profiles. Serve them in order.
    calls = {"n": 0}
    first, second = [target], profiles

    class OrderedApify(FakeApify):
        def dataset(self, dataset_id):
            if dataset_id == PROFILE:
                calls["n"] += 1
                items = first if calls["n"] == 1 else second
                return type("D", (), {"iterate_items": lambda s: iter(items)})()
            return super().dataset(dataset_id)

    main_try.ApifyClient = OrderedApify
    return profiles


def test_home_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "aktif"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "model_loaded": True, "scaler_loaded": True}


def test_health_reports_missing_assets(client, monkeypatch):
    monkeypatch.setattr(main_try, "model", None)
    monkeypatch.setattr(main_try, "ASSET_ERROR", "file not found")
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "error", "model_loaded": False, "scaler_loaded": True}
    assert "file not found" not in r.text


def test_cek_bot_503_when_assets_missing(client, monkeypatch):
    monkeypatch.setattr(main_try, "scaler", None)
    monkeypatch.setattr(main_try, "ASSET_ERROR", "secret/path/detail")
    r = client.post("/api/cek-bot", json={"username": "target"})
    assert r.status_code == 503
    assert "secret/path/detail" not in r.text


def test_private_account_returns_403(client, monkeypatch):
    setup_account(private=True)
    r = client.post("/api/cek-bot", json={"username": "@target"})
    assert r.status_code == 403
    body = r.json()
    assert body["status"] == "restricted"
    assert body["target_akun"] == "@target"


def test_no_tokens_returns_500(client, monkeypatch):
    monkeypatch.setattr(main_try, "DAFTAR_TOKEN", [])
    r = client.post("/api/cek-bot", json={"username": "target"})
    assert r.status_code == 500


def test_missing_username_is_rejected(client):
    r = client.post("/api/cek-bot", json={})
    assert r.status_code == 422


def test_unknown_account_returns_500_when_scraper_returns_nothing(client):
    FakeApify.actors = {}
    r = client.post("/api/cek-bot", json={"username": "ghost"})
    assert r.status_code == 500


def test_feature_extraction_order_and_values(client):
    setup_account(n=2)
    r = client.post("/api/cek-bot", json={"username": "target"})
    assert r.status_code == 200
    row = main_try.scaler.seen[0]
    # profile pic, digits/len username, fullname words, digits/len fullname,
    # name==username, bio length, external URL, private, posts, followers, follows
    assert row[0] == 1
    assert row[1] == pytest.approx(1 / 5)  # "user0": one digit in 5 chars
    assert row[2] == 2
    assert row[3] == pytest.approx(1 / 6)  # "User 0": one digit in 6 chars
    assert row[4] == 0
    assert row[5] == 5
    assert row[6] == 0
    assert row[7] == 0
    assert list(row[8:]) == [3, 10, 20]


@pytest.mark.parametrize(
    "bots,expected",
    [
        (0, "Sangat Berkualitas"),
        (14, "Sangat Berkualitas"),      # 14%
        (15, "Wajar"),                   # exactly 15% is NOT below 15
        (30, "Wajar"),                   # 30%
        (31, "Waspada"),                 # exactly 31% is NOT below 31
        (50, "Waspada"),                 # exactly 50% is still Waspada
        (51, "Perlu Tinjauan Lanjut"),   # 51%
        (100, "Perlu Tinjauan Lanjut"),
    ],
)
def test_decision_bands_at_boundaries(client, monkeypatch, bots, expected):
    # 100 profiles so each bot is exactly 1%.
    setup_account(n=100)
    monkeypatch.setattr(main_try, "model", FakeModel(bots=bots))
    r = client.post("/api/cek-bot", json={"username": "target"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_sampel_diperiksa"] == 100
    assert body["metrik_kualitas"]["persentase_bot"] == pytest.approx(float(bots))
    assert body["metrik_kualitas"]["persentase_asli"] == pytest.approx(100.0 - bots)
    assert body["rekomendasi"].startswith(expected)
