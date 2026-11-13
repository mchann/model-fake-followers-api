---
title: AI Fake Followers Detection
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# InsPector - AI Fake Followers Detection API

A FastAPI service that estimates what share of an Instagram account's followers look like bots. It takes a username, uses Apify scrapers to sample follower profiles, builds 11 profile features and runs them through a Keras model (`fake_followers_model.keras`) with a saved `scaler.pkl`.

## How it works

1. `POST /api/cek-bot` with `{"username": "some_account"}`.
2. The service checks the account is not private (private accounts return `403`).
3. It scrapes up to 50 followers and then their profile details (Apify actors `scraping_solutions/instagram-scraper-followers-following-no-cookies` and `apify/instagram-profile-scraper`).
4. Features per follower, in this order: has profile picture, digit ratio in username, words in full name, digit ratio in full name, name equals username, bio length, has external URL, is private, post count, follower count, following count.
5. The scaled features go to the model. A score above 0.5 counts as a bot.
6. The response gives the bot percentage of the sample and a short recommendation:

| Bot share of sample | Recommendation |
| --- | --- |
| below 15% | Sangat Berkualitas |
| 15% or more, below 31% | Wajar |
| 31% up to and including 50% | Waspada |
| more than 50% | Perlu Tinjauan Lanjut |

The result describes a sample of at most 50 followers, not every follower.

## Endpoints

- `GET /` - basic status.
- `GET /health` - `200` when the model and scaler loaded, `503` when they did not. Returns only ready/not-ready flags; the detailed load error is printed in the server log.
- `POST /api/cek-bot` - the analysis. Returns `503` if the model or scaler did not load.

Example response (illustrative values only):

```json
{
  "status": "success",
  "target_akun": "@example",
  "total_sampel_diperiksa": 50,
  "metrik_kualitas": {"persentase_bot": 12.0, "persentase_asli": 88.0, "timestamp": "2026-01-01T10:00:00+07:00"},
  "rekomendasi": "Sangat Berkualitas. ..."
}
```

## Setup

You need an Apify account. Each analysis runs paid scrapers, so use your own tokens carefully.

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in APIFY_TOKEN_1 (and more for fallback)
uvicorn main_try:app --host 0.0.0.0 --port 7860
```

Any environment variable starting with `APIFY_TOKEN_` is used. If one token fails or runs out, the next is tried. On Hugging Face, add them as Space secrets. Do not commit real tokens; `.env` is git-ignored.

Docker: `docker build -t inspector . && docker run -p 7860:7860 --env-file .env inspector`

## Tests

The tests use fake data. They do not call Apify and do not run the real model, so they check request handling, feature order, error cases and the recommendation bands, not model accuracy.

```bash
pip install -r requirements-dev.txt
pytest
```

## Notes and limits

- The tool reads public Instagram profile data through third-party scrapers. Check Instagram's terms and local privacy rules before using it on other people's accounts.
- CORS currently allows all origins and there is no authentication or rate limit.
