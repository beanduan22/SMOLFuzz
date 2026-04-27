from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


GITHUB_API = "https://api.github.com"
DEFAULT_KEYWORDS = [
    "incorrect",
    "wrong",
    "nan",
    "unexpected",
    "inconsistent",
]


@dataclass(frozen=True)
class QuerySpec:
    name: str
    repo: str
    labels: tuple[str, ...]
    extra_terms: tuple[str, ...] = ()


def github_get_json(url: str, token: str | None) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
            "User-Agent": "smolfuzz-v2-issue-study",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def search_query(repo: str, labels: tuple[str, ...], start: str, end: str, extra_terms: tuple[str, ...]) -> str:
    parts = [
        f"repo:{repo}",
        "is:issue",
        "is:closed",
        f"created:{start}..{end}",
    ]
    parts.extend(f'label:"{label}"' for label in labels)
    parts.extend(extra_terms)
    return " ".join(parts)


def search_payload(query: str, page: int, token: str | None) -> dict:
    encoded = urllib.parse.quote(query)
    url = f"{GITHUB_API}/search/issues?q={encoded}&per_page=100&page={page}"
    return github_get_json(url, token)


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _format_date(value: date) -> str:
    return value.isoformat()


def _sleep_for_search(token: str | None) -> None:
    if token:
        time.sleep(1)
    else:
        time.sleep(8)


def iter_search_items(repo: str, labels: tuple[str, ...], start: str, end: str, extra_terms: tuple[str, ...], token: str | None) -> list[dict]:
    query = search_query(repo, labels, start, end, extra_terms)
    first = search_payload(query, 1, token)
    total_count = int(first.get("total_count", 0))

    if total_count > 1000:
        start_d = _parse_date(start)
        end_d = _parse_date(end)
        if start_d >= end_d:
            raise ValueError(f"single-day query still exceeds GitHub 1000-result cap: {query}")
        mid = start_d + (end_d - start_d) // 2
        left = iter_search_items(repo, labels, _format_date(start_d), _format_date(mid), extra_terms, token)
        right = iter_search_items(repo, labels, _format_date(mid + timedelta(days=1)), _format_date(end_d), extra_terms, token)
        return left + right

    items: list[dict] = []
    batch = first.get("items", [])
    items.extend(batch)
    page = 2
    while True:
        if len(batch) < 100:
            break
        _sleep_for_search(token)
        payload = search_payload(query, page, token)
        batch = payload.get("items", [])
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def list_labels(repo: str, prefix: str, token: str | None) -> list[str]:
    labels: list[str] = []
    page = 1
    while True:
        url = f"{GITHUB_API}/repos/{repo}/labels?per_page=100&page={page}"
        payload = github_get_json(url, token)
        if not payload:
            break
        for row in payload:
            name = row["name"]
            if name.startswith(prefix):
                labels.append(name)
        if len(payload) < 100:
            break
        page += 1
        time.sleep(1)
    return sorted(labels)


def keep_issue(issue: dict, keywords: list[str]) -> bool:
    text = f"{issue.get('title', '')}\n{issue.get('body', '')}".lower()
    return any(keyword.lower() in text for keyword in keywords)


def collect_counts(specs: list[QuerySpec], start: str, end: str, keywords: list[str], token: str | None) -> dict:
    raw_hits = 0
    unique: dict[str, dict] = {}
    per_query: list[dict] = []

    for spec in specs:
        query = search_query(spec.repo, spec.labels, start, end, spec.extra_terms)
        items = iter_search_items(spec.repo, spec.labels, start, end, spec.extra_terms, token)
        raw_hits += len(items)
        for item in items:
            unique[item["html_url"]] = item
        per_query.append(
            {
                "name": spec.name,
                "repo": spec.repo,
                "labels": list(spec.labels),
                "extra_terms": list(spec.extra_terms),
                "query": query,
                "count": len(items),
            }
        )
        if not token:
            time.sleep(8)

    filtered = {
        url: issue
        for url, issue in unique.items()
        if keep_issue(issue, keywords)
    }

    return {
        "date_range": {"start": start, "end": end},
        "keywords": keywords,
        "raw_hits": raw_hits,
        "unique_hits": len(unique),
        "filtered_hits": len(filtered),
        "per_query": per_query,
        "unique_urls": sorted(unique),
        "filtered_urls": sorted(filtered),
    }


def default_specs() -> list[QuerySpec]:
    return [
        QuerySpec("pytorch_autograd", "pytorch/pytorch", ("triaged", "module: autograd")),
        QuerySpec("pytorch_functorch", "pytorch/pytorch", ("triaged", "module: functorch")),
        QuerySpec("pytorch_dynamo", "pytorch/pytorch", ("triaged", "module: dynamo")),
        QuerySpec("pytorch_ddp", "pytorch/pytorch", ("triaged", "module: ddp")),
        QuerySpec("pytorch_fsdp", "pytorch/pytorch", ("triaged", "module: fsdp")),
        QuerySpec("tensorflow_autograph", "tensorflow/tensorflow", ("type:bug", "comp:autograph")),
        QuerySpec("tensorflow_eager", "tensorflow/tensorflow", ("type:bug", "comp:eager")),
        QuerySpec("tensorflow_function", "tensorflow/tensorflow", ("type:bug", "comp:tf.function")),
        QuerySpec("tensorflow_dist_strat", "tensorflow/tensorflow", ("type:bug", "comp:dist-strat")),
        QuerySpec("tensorflow_ops", "tensorflow/tensorflow", ("type:bug", "comp:ops")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live GitHub issue counts for the SMOLFuzz v2 empirical study.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    labels_p = sub.add_parser("labels", help="List labels for a repo by prefix.")
    labels_p.add_argument("--repo", required=True, help="owner/repo")
    labels_p.add_argument("--prefix", required=True, help="Label prefix, e.g. 'module:' or 'comp:'")

    count_p = sub.add_parser("count", help="Get the GitHub Search API total_count for one label query.")
    count_p.add_argument("--repo", required=True, help="owner/repo")
    count_p.add_argument("--label", action="append", required=True, dest="labels", help="Repeat for each required label.")
    count_p.add_argument("--start", default="2021-01-01")
    count_p.add_argument("--end", default="2026-04-30")
    count_p.add_argument("--term", action="append", default=[], dest="terms", help="Optional extra raw search terms.")

    collect_p = sub.add_parser("collect", help="Collect issue counts from GitHub issue search.")
    collect_p.add_argument("--start", default="2021-01-01")
    collect_p.add_argument("--end", default="2026-04-30")
    collect_p.add_argument("--config", type=Path, default=None, help="Optional JSON file containing a list of query specs.")
    collect_p.add_argument("--out", type=Path, default=None)
    collect_p.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)

    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")

    if args.cmd == "labels":
        print(json.dumps(list_labels(args.repo, args.prefix, token), indent=2))
        return

    if args.cmd == "count":
        query = search_query(args.repo, tuple(args.labels), args.start, args.end, tuple(args.terms))
        payload = search_payload(query, 1, token)
        print(json.dumps({"query": query, "total_count": int(payload.get("total_count", 0))}, indent=2))
        return

    if args.config is None:
        specs = default_specs()
    else:
        raw_specs = json.loads(args.config.read_text())
        specs = [
            QuerySpec(
                name=row["name"],
                repo=row["repo"],
                labels=tuple(row["labels"]),
                extra_terms=tuple(row.get("extra_terms", [])),
            )
            for row in raw_specs
        ]

    result = collect_counts(specs, args.start, args.end, args.keywords, token)
    text = json.dumps(result, indent=2)
    if args.out is None:
        print(text)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
