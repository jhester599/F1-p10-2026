import importlib.util
from pathlib import Path

from src.league_digest import DigestCommentary, LeagueDigest, PlayerStanding


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "67_send_league_digest.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("league_digest_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_digest_recipients_prefers_test_override() -> None:
    script = load_script_module()

    assert script.resolve_digest_recipients(
        participant_emails=["eric@example.com", "nick@example.com"],
        test_recipient="jeffrey.r.hester@gmail.com",
    ) == ["jeffrey.r.hester@gmail.com"]


def test_resolve_digest_recipients_uses_participants_without_override() -> None:
    script = load_script_module()

    assert script.resolve_digest_recipients(
        participant_emails=["eric@example.com", "nick@example.com"],
        test_recipient=None,
    ) == ["eric@example.com", "nick@example.com"]


def test_write_dry_run_artifacts_creates_html_and_charts(tmp_path: Path) -> None:
    script = load_script_module()
    digest = LeagueDigest(
        league_label="Test League",
        sheet_round=7,
        race_name="Barcelona Grand Prix",
        p10_driver="Franco Colapinto",
        standings=[PlayerStanding("Eric", 100, 18, 1)],
        round_labels=["R7"],
        series={"Eric": [100]},
        recipients=["jeffrey.r.hester@gmail.com"],
    )
    commentary = DigestCommentary(
        headline="Headline",
        race_summary="Race summary",
        league_commentary="League commentary",
        notable_movements=["Eric led the scoring."],
    )

    artifacts = script.write_dry_run_artifacts(
        digest=digest,
        commentary=commentary,
        articles=[],
        output_dir=tmp_path,
    )

    assert artifacts["html"].exists()
    assert artifacts["standings"].exists()
    assert artifacts["series"].exists()
    assert "SPOILER WARNING" in artifacts["html"].read_text(encoding="utf-8")


def test_commentary_log_line_includes_provider_and_note() -> None:
    script = load_script_module()
    commentary = DigestCommentary(
        headline="Headline",
        race_summary="Race summary",
        league_commentary="League commentary",
        notable_movements=[],
        provider_note="Gemini commentary unavailable; used deterministic fallback.",
    )

    assert script.format_commentary_log("Test League", commentary) == (
        "Test League: commentary_provider=fallback; "
        "headline=Headline; "
        "note=Gemini commentary unavailable; used deterministic fallback."
    )


def test_fetch_race_articles_captures_search_snippets(monkeypatch) -> None:
    script = load_script_module()

    class FakeResponse:
        text = """
        <div class="result">
          <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.formula1.com%2Fen%2Flatest%2Farticle.html">F1 race report</a>
          <a class="result__snippet">Hamilton won as Colapinto claimed the final point.</a>
        </div>
        <div class="result">
          <a class="result__a" href="https://www.the-race.com/formula-1/barcelona-analysis/">The Race analysis</a>
          <a class="result__snippet">The midfield battle turned on tyre strategy.</a>
        </div>
        """

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(script.requests, "get", fake_get)

    articles = script.fetch_race_articles("Barcelona Grand Prix", 2026, limit=2)

    assert [article.source for article in articles] == ["F1.com", "The Race"]
    assert articles[0].snippet == "Hamilton won as Colapinto claimed the final point."
    assert articles[1].snippet == "The midfield battle turned on tyre strategy."


def test_fetch_race_articles_searches_priority_sources_separately(monkeypatch) -> None:
    script = load_script_module()
    requested_urls: list[str] = []

    class FakeResponse:
        text = ""

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(script.requests, "get", fake_get)

    assert script.fetch_race_articles("Barcelona Grand Prix", 2026, limit=3) == []
    assert len(requested_urls) >= 2
    assert any("site%3Aformula1.com" in url for url in requested_urls)
    assert any("site%3Athe-race.com" in url for url in requested_urls)


def test_fetch_race_articles_uses_bing_when_duckduckgo_has_no_results(monkeypatch) -> None:
    script = load_script_module()
    requested_urls: list[str] = []

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        if "duckduckgo.com" in url:
            return FakeResponse("")
        return FakeResponse(
            """
            <li class="b_algo">
              <h2><a href="https://www.formula1.com/en/latest/article/hamilton-wins.html">Hamilton wins in Barcelona</a></h2>
              <p>Hamilton claimed his first Ferrari victory as Antonelli retired late on.</p>
            </li>
            """
        )

    monkeypatch.setattr(script.requests, "get", fake_get)

    articles = script.fetch_race_articles("Barcelona Grand Prix", 2026, limit=1)

    assert any("duckduckgo.com" in url for url in requested_urls)
    assert any("bing.com" in url for url in requested_urls)
    assert articles[0].source == "F1.com"
    assert articles[0].title == "Hamilton wins in Barcelona"
    assert articles[0].snippet == "Hamilton claimed his first Ferrari victory as Antonelli retired late on."


def test_fetch_race_articles_keeps_multiple_priority_sources(monkeypatch) -> None:
    script = load_script_module()

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if "duckduckgo.com" in url:
            return FakeResponse("")
        if "site%3Aformula1.com" in url:
            return FakeResponse(
                """
                <li class="b_algo"><h2><a href="https://www.formula1.com/report-1">F1 report one</a></h2><p>F1 one.</p></li>
                <li class="b_algo"><h2><a href="https://www.formula1.com/report-2">F1 report two</a></h2><p>F1 two.</p></li>
                <li class="b_algo"><h2><a href="https://www.formula1.com/report-3">F1 report three</a></h2><p>F1 three.</p></li>
                """
            )
        return FakeResponse(
            """
            <li class="b_algo"><h2><a href="https://www.the-race.com/formula-1/report">The Race report</a></h2><p>The Race view.</p></li>
            """
        )

    monkeypatch.setattr(script.requests, "get", fake_get)

    articles = script.fetch_race_articles("Barcelona Grand Prix", 2026, limit=3)

    assert len(articles) == 3
    assert "F1.com" in [article.source for article in articles]
    assert "The Race" in [article.source for article in articles]


def test_fetch_race_articles_uses_the_race_sitemap_when_search_misses(monkeypatch) -> None:
    script = load_script_module()

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if "sitemap-posts.xml" in url:
            return FakeResponse(
                """
                <url><loc>https://www.the-race.com/formula-1/f1-2026-barcelona-grand-prix-results-as-hamilton-wins-for-ferrari/</loc></url>
                <url><loc>https://www.the-race.com/formula-1/franco-colapinto-loses-two-places-f1-barcelona-gp-post-race-penalty/</loc></url>
                """
            )
        return FakeResponse("")

    monkeypatch.setattr(script.requests, "get", fake_get)

    articles = script.fetch_race_articles("Barcelona Grand Prix", 2026, limit=2)

    assert any(article.source == "The Race" for article in articles)
    assert any("hamilton-wins-for-ferrari" in article.url for article in articles)


def test_fetch_race_articles_uses_f1_sitemap_when_search_misses(monkeypatch) -> None:
    script = load_script_module()

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if url.endswith("/en/latest/article/sitemap.xml"):
            return FakeResponse(
                '<sitemap><loc>https://www.formula1.com/en/latest/articles/sitemap-0.xml</loc></sitemap>'
            )
        if "sitemap-0.xml" in url:
            return FakeResponse(
                """
                <url><loc>https://www.formula1.com/en/latest/article/paddock-postcard-from-barcelona.abc</loc><lastmod>2026-06-15T10:00:00.000Z</lastmod></url>
                <url><loc>https://www.formula1.com/en/latest/article/hamilton-claims-stellar-maiden-grand-prix-victory-for-ferrari-in-barcelona-as-antonelli-suffers-shock-retirement.abc</loc><lastmod>2026-06-15T10:30:00.000Z</lastmod></url>
                """
            )
        return FakeResponse("")

    monkeypatch.setattr(script.requests, "get", fake_get)

    articles = script.fetch_race_articles("Barcelona Grand Prix", 2026, limit=2)

    assert any(article.source == "F1.com" for article in articles)
    assert articles[0].url.endswith("shock-retirement.abc")


def test_send_digest_email_uses_smtp_username_as_default_sender(monkeypatch, tmp_path: Path) -> None:
    script = load_script_module()
    sent_messages = []

    class FakeSMTP:
        def __init__(self, server, port):
            self.server = server
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def starttls(self):
            return None

        def login(self, username, password):
            assert username == "smtp@example.com"
            assert password == "secret"

        def send_message(self, msg):
            sent_messages.append(msg)

    attachment = tmp_path / "chart.png"
    attachment.write_bytes(b"png")
    monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "smtp@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.delenv("RESULTS_EMAIL_FROM", raising=False)
    monkeypatch.setattr(script.smtplib, "SMTP", FakeSMTP)

    assert script.send_digest_email(
        subject="Digest",
        html_body="<p>Hello</p>",
        recipients=["jeff@example.com"],
        attachments=[attachment],
    )
    assert sent_messages[0]["From"] == "smtp@example.com"
