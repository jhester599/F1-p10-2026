from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.results_automation import detect_results_table_layout


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DEFAULT_DIGEST_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_DIGEST_FALLBACK_MODEL = "gemini-2.5-flash-lite"
DEFAULT_DIGEST_LLM_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class ArticleSource:
    source: str
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class PlayerStanding:
    name: str
    total_points: int
    race_points: int
    rank: int
    prior_total: int | None = None
    prior_rank: int | None = None
    rank_delta: int | None = None


@dataclass(frozen=True)
class LeagueDigest:
    league_label: str
    sheet_round: int
    race_name: str
    p10_driver: str
    standings: list[PlayerStanding]
    round_labels: list[str]
    series: dict[str, list[int]]
    recipients: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DigestCommentary:
    headline: str
    race_summary: str
    league_commentary: str
    notable_movements: list[str]
    provider: str = "fallback"
    provider_note: str | None = None


def _as_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return 0


def extract_participant_emails(form_values: list[list[Any]]) -> list[str]:
    emails: list[str] = []
    seen: set[str] = set()
    for row in form_values[1:]:
        if len(row) < 2:
            continue
        email = str(row[1]).strip().lower()
        if not EMAIL_RE.match(email) or email in seen:
            continue
        seen.add(email)
        emails.append(email)
    return emails


def _rank_by_total(values: dict[str, int]) -> dict[str, int]:
    ranked = sorted(values.items(), key=lambda item: (-item[1], item[0].lower()))
    return {name: index for index, (name, _value) in enumerate(ranked, start=1)}


def _row_at(values: list[list[Any]], one_based_row: int) -> list[Any]:
    index = one_based_row - 1
    return values[index] if 0 <= index < len(values) else []


def _value_at(row: list[Any], one_based_col: int) -> Any:
    index = one_based_col - 1
    return row[index] if 0 <= index < len(row) else ""


def parse_league_digest(
    *,
    results_values: list[list[Any]],
    sheet_round: int,
    race_name: str,
    p10_driver: str,
    league_label: str = "F1 P10 League",
    recipients: list[str] | None = None,
) -> LeagueDigest:
    layout = detect_results_table_layout(results_values)
    player_count = layout["player_count"]
    header = results_values[0] if results_values else []
    total_row = _row_at(results_values, 2)
    race_row = _row_at(results_values, sheet_round + 2)
    prior_series_row = _row_at(results_values, sheet_round + 1)

    names = [str(_value_at(header, col)).strip() for col in range(2, player_count + 2)]
    totals = {name: _as_int(_value_at(total_row, col)) for col, name in enumerate(names, start=2) if name}
    race_points = {
        name: _as_int(_value_at(race_row, col))
        for col, name in enumerate(names, start=2)
        if name
    }

    prior_totals: dict[str, int] = {}
    series_start = layout["series_start_column"]
    for player_index, name in enumerate(names):
        if not name:
            continue
        prior_totals[name] = _as_int(_value_at(prior_series_row, series_start + player_index + 1))

    current_ranks = _rank_by_total(totals)
    prior_ranks = _rank_by_total(prior_totals) if any(prior_totals.values()) else {}

    standings = [
        PlayerStanding(
            name=name,
            total_points=totals[name],
            race_points=race_points.get(name, 0),
            rank=current_ranks[name],
            prior_total=prior_totals.get(name),
            prior_rank=prior_ranks.get(name),
            rank_delta=(
                prior_ranks[name] - current_ranks[name]
                if name in prior_ranks
                else None
            ),
        )
        for name in totals
    ]
    standings.sort(key=lambda row: (row.rank, row.name.lower()))

    round_labels: list[str] = []
    series: dict[str, list[int]] = {name: [] for name in totals}
    for row in results_values[2:]:
        label = str(_value_at(row, series_start)).strip() or str(_value_at(row, 1)).strip()
        if not label:
            continue
        player_values = [
            _as_int(_value_at(row, series_start + player_index + 1))
            for player_index in range(player_count)
        ]
        if not any(player_values):
            continue
        round_labels.append(label)
        for name, value in zip(names, player_values):
            if name:
                series[name].append(value)

    return LeagueDigest(
        league_label=league_label,
        sheet_round=sheet_round,
        race_name=race_name,
        p10_driver=p10_driver,
        standings=standings,
        round_labels=round_labels,
        series=series,
        recipients=recipients or [],
    )


def build_spoiler_subject(digest: LeagueDigest) -> str:
    return f"[F1 P10 SPOILERS] {digest.league_label} - R{digest.sheet_round} {digest.race_name} results"


def build_fallback_commentary(digest: LeagueDigest) -> DigestCommentary:
    leader = digest.standings[0] if digest.standings else None
    round_best = max(digest.standings, key=lambda row: (row.race_points, -row.rank), default=None)
    headline = f"{digest.race_name}: P10 drama lands in the league table"
    race_summary = (
        f"The official P10 finisher was {digest.p10_driver}. "
        f"Round {digest.sheet_round} points have been added to the league standings."
    )
    league_commentary = (
        f"{leader.name} leads on {leader.total_points} total points."
        if leader
        else "League totals are updated."
    )
    notable: list[str] = []
    if round_best:
        notable.append(f"{round_best.name} led the round with {round_best.race_points} points.")
    movers = [row for row in digest.standings if row.rank_delta and row.rank_delta > 0]
    if movers:
        top_mover = max(movers, key=lambda row: (row.rank_delta or 0, row.race_points))
        notable.append(f"{top_mover.name} gained {top_mover.rank_delta} ranking spot(s).")
    return DigestCommentary(
        headline=headline,
        race_summary=race_summary,
        league_commentary=league_commentary,
        notable_movements=notable,
        provider="fallback",
    )


def _commentary_prompt(digest: LeagueDigest, articles: list[ArticleSource]) -> str:
    payload = {
        "race": {
            "round": digest.sheet_round,
            "name": digest.race_name,
            "p10_driver": digest.p10_driver,
        },
        "standings": [row.__dict__ for row in digest.standings],
        "articles": [article.__dict__ for article in articles],
        "style": (
            "Write lively but concise F1 fantasy league commentary. "
            "Mention spoilers are already disclosed by the email header. "
            "Ground the race_summary in the supplied source titles and snippets when available. "
            "Do not invent facts beyond the supplied race, league, and source data. "
            "Return only one JSON object with string keys "
            '"headline", "race_summary", "league_commentary", and array key "notable_movements".'
        ),
    }
    return json.dumps(payload, ensure_ascii=True)


def _parse_commentary_json(raw: str) -> DigestCommentary:
    data = json.loads(raw)
    return DigestCommentary(
        headline=str(data.get("headline", "")).strip(),
        race_summary=str(data.get("race_summary", "")).strip(),
        league_commentary=str(data.get("league_commentary", "")).strip(),
        notable_movements=[str(item).strip() for item in data.get("notable_movements", []) if str(item).strip()],
        provider="gemini",
    )


def summarize_gemini_error(exc: Exception, max_length: int = 320) -> str:
    message = " ".join(str(exc).split())
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(env_name)
        if secret:
            message = message.replace(secret, f"[REDACTED_{env_name}]")
    summary = f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__
    if len(summary) > max_length:
        return summary[: max_length - 3].rstrip() + "..."
    return summary


def digest_llm_timeout_ms() -> int:
    raw = os.getenv("RESULTS_DIGEST_LLM_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_DIGEST_LLM_TIMEOUT_SECONDS * 1000
    try:
        seconds = int(raw)
    except ValueError:
        return DEFAULT_DIGEST_LLM_TIMEOUT_SECONDS * 1000
    return max(1, seconds) * 1000


def digest_model_sequence(model: str | None = None) -> list[str]:
    explicit_model = model or os.getenv("RESULTS_DIGEST_MODEL", "").strip()
    if explicit_model:
        return [explicit_model]
    return [DEFAULT_DIGEST_MODEL, DEFAULT_DIGEST_FALLBACK_MODEL]


def is_gemma_model(model_name: str) -> bool:
    return model_name.lower().startswith("gemma-")


def commentary_generation_config(model_name: str) -> dict[str, Any]:
    config: dict[str, Any] = {"response_mime_type": "application/json"}
    if is_gemma_model(model_name):
        return config
    config["response_schema"] = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "race_summary": {"type": "string"},
            "league_commentary": {"type": "string"},
            "notable_movements": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["headline", "race_summary", "league_commentary", "notable_movements"],
    }
    return config


def request_gemini_commentary(
    digest: LeagueDigest,
    articles: list[ArticleSource],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> DigestCommentary:
    model_names = digest_model_sequence(model)
    last_error: Exception | None = None
    try:
        if client is None:
            if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is not configured")
            from google import genai
            from google.genai import types

            client = genai.Client(
                http_options=types.HttpOptions(timeout=digest_llm_timeout_ms())
            )
        for model_name in model_names:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=_commentary_prompt(digest, articles),
                    config=commentary_generation_config(model_name),
                )
                commentary = _parse_commentary_json(response.text)
                if commentary.headline and commentary.race_summary and commentary.league_commentary:
                    return DigestCommentary(
                        headline=commentary.headline,
                        race_summary=commentary.race_summary,
                        league_commentary=commentary.league_commentary,
                        notable_movements=commentary.notable_movements,
                        provider="gemini",
                        provider_note=f"Gemini model {model_name} generated this commentary.",
                    )
                raise ValueError("Gemini response omitted required prose")
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("No Gemini models configured")
    except Exception as exc:
        fallback = build_fallback_commentary(digest)
        return DigestCommentary(
            headline=fallback.headline,
            race_summary=fallback.race_summary,
            league_commentary=fallback.league_commentary,
            notable_movements=fallback.notable_movements,
            provider="fallback",
            provider_note=(
                "Gemini commentary unavailable; used deterministic fallback. "
                f"Gemini error: {summarize_gemini_error(exc)}"
            ),
        )


def render_digest_html(
    digest: LeagueDigest,
    commentary: DigestCommentary,
    articles: list[ArticleSource],
) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{standing.rank}</td>"
        f"<td>{html.escape(standing.name)}</td>"
        f"<td>{standing.total_points}</td>"
        f"<td>{standing.race_points}</td>"
        f"<td>{'+' if standing.rank_delta and standing.rank_delta > 0 else ''}{standing.rank_delta or 0}</td>"
        "</tr>"
        for standing in digest.standings
    )
    movement_items = "\n".join(f"<li>{html.escape(item)}</li>" for item in commentary.notable_movements)
    source_items = "\n".join(
        f'<li><a href="{html.escape(article.url)}">{html.escape(article.source)}: {html.escape(article.title)}</a></li>'
        for article in articles
    )
    provider_note = f"<p><em>{html.escape(commentary.provider_note)}</em></p>" if commentary.provider_note else ""
    return f"""<!doctype html>
<html>
<body>
  <h1 style="color:#b00020;">SPOILER WARNING: F1 race and league results below</h1>
  <h2>{html.escape(digest.league_label)} - R{digest.sheet_round} {html.escape(digest.race_name)}</h2>
  <h3>{html.escape(commentary.headline)}</h3>
  <p>{html.escape(commentary.race_summary)}</p>
  <p><strong>P10:</strong> {html.escape(digest.p10_driver)}</p>
  <p>{html.escape(commentary.league_commentary)}</p>
  {provider_note}
  <h3>Notable League Notes</h3>
  <ul>{movement_items}</ul>
  <h3>Standings</h3>
  <table border="1" cellpadding="6" cellspacing="0">
    <thead><tr><th>Rank</th><th>Player</th><th>Total</th><th>Race Points</th><th>Rank +/-</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p>Charts are attached: total standings and score progression by round.</p>
  <h3>Race Sources</h3>
  <ul>{source_items}</ul>
</body>
</html>
"""


def render_league_charts(digest: LeagueDigest, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    standings_path = output_dir / f"league_R{digest.sheet_round:02d}_standings.png"
    series_path = output_dir / f"league_R{digest.sheet_round:02d}_series.png"

    names = [row.name for row in digest.standings]
    totals = [row.total_points for row in digest.standings]
    plt.figure(figsize=(10, 5))
    plt.bar(names, totals, color="#d62728")
    plt.title(f"{digest.league_label} Total Points")
    plt.ylabel("Points")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(standings_path, dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    for name, values in digest.series.items():
        if values:
            plt.plot(digest.round_labels[-len(values):], values, marker="o", label=name)
    plt.title(f"{digest.league_label} Points By Round")
    plt.ylabel("Total Points")
    plt.xlabel("Round")
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(series_path, dpi=140)
    plt.close()

    return {"standings": standings_path, "series": series_path}
