from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


GITHUB_API = "https://api.github.com"
USER_AGENT = "smolfuzz-unfixed-bug-miner"


@dataclass(frozen=True)
class CategoryQuery:
    name: str
    repo: str
    terms: tuple[str, ...]
    description: str


@dataclass
class IssueRecord:
    repo: str
    number: int
    title: str
    url: str
    state: str
    labels: list[str]
    created_at: str
    updated_at: str
    comments: int
    reactions: int
    body_excerpt: str
    categories: list[str] = field(default_factory=list)
    apis: list[str] = field(default_factory=list)
    has_repro: bool = False


_PYTORCH_API_RE = re.compile(r"\btorch(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_TF_API_RE = re.compile(r"\btf(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_KERAS_API_RE = re.compile(r"\b(?:tf\.keras|keras)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_REPRO_HINTS = ("```python", "```py", "minimal repro", "reproduc", "expected behavior")


PYTORCH_CATEGORIES: tuple[CategoryQuery, ...] = (
    CategoryQuery(
        name="cpu_gpu_mismatch",
        repo="pytorch/pytorch",
        terms=("cpu", "cuda", "mismatch"),
        description="CPU vs CUDA/MPS numeric divergence",
    ),
    CategoryQuery(
        name="jit_eager_mismatch",
        repo="pytorch/pytorch",
        terms=("torch.compile", "eager", "mismatch"),
        description="torch.compile / torch.jit vs eager divergence",
    ),
    CategoryQuery(
        name="gradient_mismatch",
        repo="pytorch/pytorch",
        terms=("gradcheck", "gradient", "incorrect"),
        description="Autograd / backward incorrect gradients",
    ),
    CategoryQuery(
        name="distributed_mismatch",
        repo="pytorch/pytorch",
        terms=("DDP", "incorrect"),
        description="Distributed training divergence",
    ),
    CategoryQuery(
        name="numerical_accuracy",
        repo="pytorch/pytorch",
        terms=("incorrect", "result", "wrong"),
        description="Numerical accuracy / wrong-result regressions",
    ),
    CategoryQuery(
        name="nan_inf",
        repo="pytorch/pytorch",
        terms=("nan", "unexpected"),
        description="Unexpected NaN / Inf produced by an op",
    ),
)


TENSORFLOW_CATEGORIES: tuple[CategoryQuery, ...] = (
    CategoryQuery(
        name="cpu_gpu_mismatch",
        repo="tensorflow/tensorflow",
        terms=("cpu", "gpu", "inconsistent"),
        description="CPU vs GPU divergence",
    ),
    CategoryQuery(
        name="tf_function_mismatch",
        repo="tensorflow/tensorflow",
        terms=("tf.function", "eager", "mismatch"),
        description="tf.function / XLA vs eager divergence",
    ),
    CategoryQuery(
        name="gradient_mismatch",
        repo="tensorflow/tensorflow",
        terms=("gradient", "incorrect"),
        description="GradientTape incorrect gradients",
    ),
    CategoryQuery(
        name="distribute_mismatch",
        repo="tensorflow/tensorflow",
        terms=("MirroredStrategy", "incorrect"),
        description="tf.distribute divergence",
    ),
    CategoryQuery(
        name="numerical_accuracy",
        repo="tensorflow/tensorflow",
        terms=("incorrect", "result"),
        description="Numerical accuracy / wrong-result regressions",
    ),
    CategoryQuery(
        name="nan_inf",
        repo="tensorflow/tensorflow",
        terms=("nan", "unexpected"),
        description="Unexpected NaN / Inf produced by an op",
    ),
)


def _build_query(cq: CategoryQuery) -> str:
    parts = [f"repo:{cq.repo}", "is:issue", "is:open"]
    parts.extend(cq.terms)
    return " ".join(parts)


def _http_get(url: str, token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining is not None and int(remaining) <= 1:
                    reset = resp.headers.get("X-RateLimit-Reset")
                    sleep_s = max(2, int(reset) - int(time.time()) + 2) if reset else 60
                    sys.stderr.write(f"[ratelimit] sleeping {sleep_s}s\n")
                    time.sleep(sleep_s)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                wait = 60 * (attempt + 1)
                sys.stderr.write(f"[backoff] {exc.code} → sleep {wait}s\n")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET failed after retries: {url}")


def _search_issues(query: str, page: int, token: str | None) -> dict:
    encoded = urllib.parse.quote(query)
    url = f"{GITHUB_API}/search/issues?q={encoded}&per_page=100&page={page}&sort=updated&order=desc"
    return _http_get(url, token)


def _extract_apis(text: str, framework: str) -> list[str]:
    found: set[str] = set()
    if framework == "pytorch":
        found.update(_PYTORCH_API_RE.findall(text))
    else:
        found.update(_TF_API_RE.findall(text))
        found.update(_KERAS_API_RE.findall(text))
    cleaned = {api.rstrip(".") for api in found if api.count(".") >= 1}
    return sorted(cleaned)


def _detect_repro(text: str) -> bool:
    low = text.lower()
    return any(hint in low for hint in _REPRO_HINTS)


def _categorize_repo(repo: str) -> str:
    return "pytorch" if "pytorch" in repo else "tensorflow"


def _to_record(item: dict, category: str) -> IssueRecord:
    repo_full = item.get("repository_url", "").rsplit("/", 2)[-2:]
    repo = "/".join(repo_full) if len(repo_full) == 2 else ""
    framework = _categorize_repo(repo)
    body = item.get("body") or ""
    title = item.get("title") or ""
    text = f"{title}\n{body}"
    apis = _extract_apis(text, framework)
    excerpt = body.strip().splitlines()
    excerpt_short = "\n".join(excerpt[:8])[:600]
    reactions = (item.get("reactions") or {}).get("total_count", 0)
    return IssueRecord(
        repo=repo,
        number=int(item.get("number", 0)),
        title=title,
        url=item.get("html_url", ""),
        state=item.get("state", ""),
        labels=[lbl.get("name", "") for lbl in item.get("labels", [])],
        created_at=item.get("created_at", ""),
        updated_at=item.get("updated_at", ""),
        comments=int(item.get("comments", 0)),
        reactions=int(reactions),
        body_excerpt=excerpt_short,
        categories=[category],
        apis=apis,
        has_repro=_detect_repro(text),
    )


def collect(
    queries: Iterable[CategoryQuery],
    max_per_query: int,
    token: str | None,
    sleep_s: float,
) -> list[IssueRecord]:
    by_url: dict[str, IssueRecord] = {}
    for cq in queries:
        query = _build_query(cq)
        sys.stderr.write(f"[query] {cq.name} ({cq.repo}): {query}\n")
        fetched = 0
        page = 1
        while fetched < max_per_query:
            payload = _search_issues(query, page, token)
            items = payload.get("items", [])
            if not items:
                break
            for item in items:
                record = _to_record(item, cq.name)
                existing = by_url.get(record.url)
                if existing is None:
                    by_url[record.url] = record
                else:
                    if cq.name not in existing.categories:
                        existing.categories.append(cq.name)
                fetched += 1
                if fetched >= max_per_query:
                    break
            if len(items) < 100:
                break
            page += 1
            time.sleep(sleep_s)
        time.sleep(sleep_s)
    return sorted(by_url.values(), key=lambda r: (r.repo, -r.reactions, -r.comments))


def to_jsonable(records: list[IssueRecord]) -> list[dict]:
    return [
        {
            "repo": r.repo,
            "number": r.number,
            "title": r.title,
            "url": r.url,
            "state": r.state,
            "labels": r.labels,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "comments": r.comments,
            "reactions": r.reactions,
            "categories": r.categories,
            "apis": r.apis,
            "has_repro": r.has_repro,
            "body_excerpt": r.body_excerpt,
        }
        for r in records
    ]


def summarize(records: list[IssueRecord]) -> dict:
    by_repo: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_label: dict[str, int] = {}
    api_counter: dict[str, int] = {}
    repro_count = 0
    for r in records:
        by_repo[r.repo] = by_repo.get(r.repo, 0) + 1
        for cat in r.categories:
            by_category[cat] = by_category.get(cat, 0) + 1
        for lbl in r.labels:
            by_label[lbl] = by_label.get(lbl, 0) + 1
        for api in r.apis:
            api_counter[api] = api_counter.get(api, 0) + 1
        if r.has_repro:
            repro_count += 1
    top_apis = sorted(api_counter.items(), key=lambda x: -x[1])[:30]
    top_labels = sorted(by_label.items(), key=lambda x: -x[1])[:20]
    return {
        "total_issues": len(records),
        "with_repro": repro_count,
        "by_repo": by_repo,
        "by_category": by_category,
        "top_labels": top_labels,
        "top_apis": top_apis,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine open / unfixed bugs from pytorch/pytorch and tensorflow/tensorflow "
                    "and emit a structured catalog usable as fuzzer seeds.",
    )
    p.add_argument(
        "--repo",
        choices=("pytorch", "tensorflow", "both"),
        default="both",
        help="Which framework to mine.",
    )
    p.add_argument(
        "--category",
        action="append",
        default=[],
        help="Restrict to specific categories (e.g. cpu_gpu_mismatch). Repeat. Default: all.",
    )
    p.add_argument("--max-per-query", type=int, default=100)
    p.add_argument(
        "--sleep",
        type=float,
        default=8.0,
        help="Seconds between requests. 8 is safe unauthenticated; with GITHUB_TOKEN you can drop to ~1.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "unfixed_bugs.json",
    )
    p.add_argument("--summary-only", action="store_true", help="Print summary only, do not write catalog.")
    return p.parse_args()


def select_queries(repo_choice: str, categories: list[str]) -> list[CategoryQuery]:
    pool: list[CategoryQuery] = []
    if repo_choice in ("pytorch", "both"):
        pool.extend(PYTORCH_CATEGORIES)
    if repo_choice in ("tensorflow", "both"):
        pool.extend(TENSORFLOW_CATEGORIES)
    if categories:
        wanted = set(categories)
        pool = [q for q in pool if q.name in wanted]
    return pool


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.stderr.write("[note] GITHUB_TOKEN not set — using unauthenticated rate limits (60/hr).\n")
    queries = select_queries(args.repo, args.category)
    if not queries:
        sys.stderr.write("[error] no matching queries\n")
        return 2
    records = collect(queries, args.max_per_query, token, args.sleep)
    payload = {
        "summary": summarize(records),
        "issues": to_jsonable(records),
    }
    if args.summary_only:
        json.dump(payload["summary"], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    sys.stderr.write(f"[done] {len(records)} issues → {args.out}\n")
    json.dump(payload["summary"], sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
