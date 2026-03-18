"""Tests for internet search client fallbacks and parsing."""

import json

from drug_discovery.web_scraping.scraper import InternetSearchClient


class _FakeResponse:
    def __init__(self, text="", json_data=None):
        self.text = text
        self._json_data = json_data or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


def test_google_search_path(monkeypatch):
    client = InternetSearchClient(google_api_key="k", google_cse_id="cx")

    def _fake_get(url, params=None, timeout=0, headers=None):
        assert "googleapis" in url
        return _FakeResponse(
            json_data={
                "items": [
                    {"title": "Paper", "link": "https://example.org/paper", "snippet": "summary"},
                ]
            }
        )

    monkeypatch.setattr("drug_discovery.web_scraping.scraper.requests.get", _fake_get)

    results = client.search_web("egfr synthesis", max_results=3, prefer_google=True)
    assert len(results) == 1
    assert results[0]["source"] == "google-cse"


def test_duckduckgo_fallback_path(monkeypatch):
    client = InternetSearchClient()

    html = (
        '<a class="result__a" href="https://example.org/a">Example A</a>'
        '<a class="result__a" href="https://example.org/b">Example B</a>'
    )

    def _fake_get(url, params=None, timeout=0, headers=None):
        assert "duckduckgo.com" in url
        return _FakeResponse(text=html)

    monkeypatch.setattr("drug_discovery.web_scraping.scraper.requests.get", _fake_get)

    results = client.search_web("egfr synthesis", max_results=2, prefer_google=False)
    assert len(results) == 2
    assert results[0]["source"] == "duckduckgo"


def test_go_backend_path(monkeypatch):
    client = InternetSearchClient(go_search_bin="/usr/local/bin/zane-fastsearch")

    class _Completed:
        def __init__(self):
            self.stdout = json.dumps(
                [
                    {
                        "title": "Go Result",
                        "url": "https://example.org/go",
                        "snippet": "from go",
                        "source": "go-fastsearch",
                    }
                ]
            )

    def _fake_run(command, capture_output=False, text=False, timeout=0, check=False):
        assert command[0] == "/usr/local/bin/zane-fastsearch"
        return _Completed()

    monkeypatch.setattr("drug_discovery.web_scraping.scraper.subprocess.run", _fake_run)

    results = client.search_web("egfr synthesis", max_results=3, prefer_google=False)
    assert len(results) == 1
    assert results[0]["source"] == "go-fastsearch"
