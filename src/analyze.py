"""
Issue Analysis & Text Intelligence
====================================
Loads an ingested issue dataset, profiles the corpus, extracts keywords
via TF-IDF, and applies heuristic categorisation.

All computation uses the Python standard library only.

Usage:
    python -m src.analyze data/raw/fastapi_fastapi_issues_*.json
    python -m src.analyze data/raw/fastapi_fastapi_issues_*.json --top-keywords 10 --save
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ANALYZED_DIR = Path(__file__).resolve().parent.parent / "data" / "analyzed"

# ---------------------------------------------------------------------------
# Stopwords — compact English set covering most noise
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    "a about above after again against all am an and any are aren't as at be "
    "because been before being below between both but by can can't cannot could "
    "couldn't did didn't do does doesn't doing don't down during each few for "
    "from further get got had hadn't has hasn't have haven't having he he'd "
    "he'll he's her here here's hers herself him himself his how how's i i'd "
    "i'll i'm i've if in into is isn't it it's its itself just let's me might "
    "more most mustn't my myself no nor not of off on once only or other ought "
    "our ours ourselves out over own re s same shan't she she'd she'll she's "
    "should shouldn't so some such t than that that's the their theirs them "
    "themselves then there there's these they they'd they'll they're they've "
    "this those through to too under until up upon us ve very was wasn't we "
    "we'd we'll we're we've were weren't what what's when when's where where's "
    "which while who who's whom why why's will with won't would wouldn't x "
    "you you'd you'll you're you've your yours yourself yourselves "
    # Tech/GitHub noise
    "use using used also like would could should need want "
    "one two get make way may new see well even still already "
    "just really try tried thing things going got know work works working "
    "http https www com github com org io api json html "
    "issue pr pull request please thank thanks hi hello".split()
)

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

# Regex to strip markdown fences, inline code, URLs, HTML tags
_MD_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_URL_RE = re.compile(r"https?://\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def preprocess_text(text: str) -> str:
    """Strip markdown, URLs, HTML, and non-alpha characters; lowercase."""
    text = _MD_CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Preprocess and tokenize text into a list of non-stopword tokens."""
    cleaned = preprocess_text(text)
    tokens = cleaned.split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


# ---------------------------------------------------------------------------
# TF-IDF (standard library implementation)
# ---------------------------------------------------------------------------

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total tokens in document."""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """
    Inverse document frequency with smoothing:
        idf(t) = log((1 + N) / (1 + df(t))) + 1
    where N = total docs, df(t) = docs containing term t.
    """
    n = len(corpus_tokens)
    df: Counter[str] = Counter()
    for doc_tokens in corpus_tokens:
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            df[term] += 1

    return {
        term: math.log((1 + n) / (1 + freq)) + 1
        for term, freq in df.items()
    }


def compute_tfidf(
    corpus_tokens: list[list[str]],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Compute TF-IDF scores for each document in the corpus.

    Returns
    -------
    (tfidf_per_doc, idf_scores)
        tfidf_per_doc: list of {term: tfidf_score} dicts, one per document
        idf_scores: global IDF values
    """
    idf = compute_idf(corpus_tokens)
    tfidf_docs: list[dict[str, float]] = []

    for doc_tokens in corpus_tokens:
        tf = compute_tf(doc_tokens)
        tfidf = {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
        tfidf_docs.append(tfidf)

    return tfidf_docs, idf


def top_keywords(tfidf: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    """Return the top-n keywords by TF-IDF score from a single document."""
    return sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:n]


def corpus_top_keywords(
    tfidf_docs: list[dict[str, float]], n: int = 20
) -> list[tuple[str, float]]:
    """Aggregate TF-IDF across all documents and return top-n corpus keywords."""
    aggregated: Counter[str] = Counter()
    for doc_tfidf in tfidf_docs:
        for term, score in doc_tfidf.items():
            aggregated[term] += score
    return aggregated.most_common(n)


# ---------------------------------------------------------------------------
# Heuristic categorisation
# ---------------------------------------------------------------------------

# Patterns are (compiled_regex, category, weight)
# The category with the highest total weight wins.
_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    # Bug indicators
    (re.compile(r"\berror\b", re.I), "bug", 2.0),
    (re.compile(r"\bbug\b", re.I), "bug", 3.0),
    (re.compile(r"\bcrash(?:es|ed|ing)?\b", re.I), "bug", 3.0),
    (re.compile(r"\bfail(?:s|ed|ing|ure)?\b", re.I), "bug", 2.0),
    (re.compile(r"\btraceback\b", re.I), "bug", 3.0),
    (re.compile(r"\bexception\b", re.I), "bug", 2.0),
    (re.compile(r"\bunexpected\b", re.I), "bug", 1.5),
    (re.compile(r"\b(?:broken|breaks|break)\b", re.I), "bug", 2.0),
    (re.compile(r"\bregression\b", re.I), "bug", 3.0),
    (re.compile(r"\b500\b", re.I), "bug", 2.0),
    (re.compile(r"\binternal server error\b", re.I), "bug", 3.0),
    (re.compile(r"\bsegfault\b", re.I), "bug", 3.0),
    (re.compile(r"\bnot work(?:ing|s)?\b", re.I), "bug", 2.0),
    (re.compile(r"\bdoesn'?t work\b", re.I), "bug", 2.0),
    (re.compile(r"\binstead of\b", re.I), "bug", 1.0),

    # Feature / enhancement indicators
    (re.compile(r"\bfeature\b", re.I), "feature", 3.0),
    (re.compile(r"\benhance(?:ment)?\b", re.I), "feature", 3.0),
    (re.compile(r"\badd(?:ing)?\s+(?:support|ability|option)\b", re.I), "feature", 2.5),
    (re.compile(r"\bproposal\b", re.I), "feature", 2.5),
    (re.compile(r"\bwould be (?:nice|great|useful)\b", re.I), "feature", 2.0),
    (re.compile(r"\bshould (?:support|allow|enable|have)\b", re.I), "feature", 1.5),
    (re.compile(r"\bwish\b", re.I), "feature", 1.5),
    (re.compile(r"\bimprove(?:ment)?\b", re.I), "feature", 1.5),
    (re.compile(r"\bnew\s+(?:feature|option|flag|parameter)\b", re.I), "feature", 2.5),

    # Question / support indicators
    (re.compile(r"\bhow (?:to|do|can|does)\b", re.I), "question", 3.0),
    (re.compile(r"\bquestion\b", re.I), "question", 3.0),
    (re.compile(r"\bis it possible\b", re.I), "question", 2.5),
    (re.compile(r"\bcan (?:i|we|you)\b", re.I), "question", 1.5),
    (re.compile(r"\bhelp\b", re.I), "question", 1.0),
    (re.compile(r"\bwhat is\b", re.I), "question", 2.0),
    (re.compile(r"\bwhy does\b", re.I), "question", 2.0),
    (re.compile(r"\banyone\b", re.I), "question", 1.0),
    (re.compile(r"\?\s*$", re.M), "question", 1.5),

    # Documentation indicators
    (re.compile(r"\bdoc(?:s|umentation)?\b", re.I), "docs", 3.0),
    (re.compile(r"\btypo\b", re.I), "docs", 2.0),
    (re.compile(r"\breadme\b", re.I), "docs", 2.0),
    (re.compile(r"\bexample\b", re.I), "docs", 1.0),
    (re.compile(r"\btutorial\b", re.I), "docs", 2.0),
    (re.compile(r"\bmissing\s+(?:docs|documentation|example)\b", re.I), "docs", 3.0),
]


def categorize_issue(title: str, body: str) -> tuple[str, dict[str, float]]:
    """
    Categorise an issue by weighted pattern matching on title + body.

    The title is weighted 2× since it's a more intentional signal.

    Returns
    -------
    (category, scores)
        category: "bug" | "feature" | "question" | "docs" | "other"
        scores: {category: total_weight} for transparency
    """
    scores: dict[str, float] = {}
    text_parts = [(title, 2.0), (body[:3000], 1.0)]  # Cap body scan at 3k chars

    for text, text_weight in text_parts:
        for pattern, category, weight in _CATEGORY_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                scores[category] = scores.get(category, 0) + weight * text_weight * len(matches)

    if not scores:
        return "other", scores

    best = max(scores, key=lambda k: scores[k])

    # Require a minimum confidence gap — if top two are very close, mark "other"
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 1.0:
        # Ambiguous — still return the top category but flag it
        pass

    return best, scores


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSON issue dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def find_latest_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Find the most recently created dataset file in data/raw/."""
    files = sorted(data_dir.glob("*_issues_*.json"))
    if not files:
        raise FileNotFoundError(f"No dataset files found in {data_dir}")
    return files[-1]


# ---------------------------------------------------------------------------
# Corpus profiling
# ---------------------------------------------------------------------------

def profile_corpus(issues: list[dict[str, Any]], top_n_keywords: int = 15) -> dict[str, Any]:
    """
    Run full analysis on the issue corpus.

    Returns a rich profile dict containing:
    - basic stats
    - text length distributions
    - category distribution
    - top TF-IDF keywords (corpus-level and per-category)
    - per-issue enrichments (category, keywords)
    """
    total = len(issues)
    if total == 0:
        return {"total": 0}

    # --- Basic stats ---
    open_count = sum(1 for i in issues if i["state"] == "open")
    closed_count = sum(1 for i in issues if i["state"] == "closed")
    with_body = sum(1 for i in issues if i.get("body"))
    with_labels = sum(1 for i in issues if i.get("labels"))
    total_comments = sum(i.get("comment_count", 0) for i in issues)

    # --- Text length distributions ---
    title_lengths = [len(i.get("title", "")) for i in issues]
    body_lengths = [len(i.get("body", "")) for i in issues]
    title_word_counts = [len(i.get("title", "").split()) for i in issues]
    body_word_counts = [len(i.get("body", "").split()) for i in issues]

    def _stats(values: list[int | float]) -> dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        s = sorted(values)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "mean": round(sum(s) / n, 1),
            "median": s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 1),
        }

    # --- Tokenise corpus ---
    corpus_tokens: list[list[str]] = []
    for issue in issues:
        combined = (issue.get("title", "") + " " + issue.get("body", ""))
        tokens = tokenize(combined)
        corpus_tokens.append(tokens)

    # --- TF-IDF ---
    tfidf_docs, idf_scores = compute_tfidf(corpus_tokens)
    corpus_kws = corpus_top_keywords(tfidf_docs, n=top_n_keywords)

    # --- Categorise each issue ---
    categories: list[str] = []
    enriched_issues: list[dict[str, Any]] = []

    for i, issue in enumerate(issues):
        cat, cat_scores = categorize_issue(
            issue.get("title", ""), issue.get("body", "")
        )
        kws = top_keywords(tfidf_docs[i], n=8)
        categories.append(cat)

        enriched = dict(issue)
        enriched["_category"] = cat
        enriched["_category_scores"] = {k: round(v, 2) for k, v in cat_scores.items()}
        enriched["_keywords"] = [{"term": t, "score": round(s, 4)} for t, s in kws]
        enriched_issues.append(enriched)

    cat_dist = Counter(categories)

    # --- Per-category keyword analysis ---
    cat_keywords: dict[str, list[tuple[str, float]]] = {}
    for cat in sorted(cat_dist.keys()):
        cat_tfidf = [tfidf_docs[i] for i, c in enumerate(categories) if c == cat]
        if cat_tfidf:
            cat_keywords[cat] = corpus_top_keywords(cat_tfidf, n=top_n_keywords)

    # --- Label vs category comparison ---
    label_counter: Counter[str] = Counter()
    for issue in issues:
        for lbl in issue.get("labels", []):
            label_counter[lbl] += 1

    return {
        "total": total,
        "open": open_count,
        "closed": closed_count,
        "with_body": with_body,
        "with_labels": with_labels,
        "total_comments": total_comments,
        "title_length_stats": _stats(title_lengths),
        "body_length_stats": _stats(body_lengths),
        "title_word_stats": _stats(title_word_counts),
        "body_word_stats": _stats(body_word_counts),
        "category_distribution": dict(cat_dist.most_common()),
        "corpus_keywords": [(t, round(s, 4)) for t, s in corpus_kws],
        "category_keywords": {
            cat: [(t, round(s, 4)) for t, s in kws]
            for cat, kws in cat_keywords.items()
        },
        "label_distribution": dict(label_counter.most_common()),
        "enriched_issues": enriched_issues,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(profile: dict[str, Any]) -> None:
    """Print a formatted analysis report to stdout."""
    total = profile.get("total", 0)
    if total == 0:
        print("Empty dataset — nothing to analyse.")
        return

    print("\n" + "=" * 70)
    print("  ISSUE CORPUS ANALYSIS REPORT")
    print("=" * 70)

    # Basic stats
    print(f"\n  Total issues       : {total}")
    print(f"  Open / Closed      : {profile['open']} / {profile['closed']}")
    print(f"  With body text     : {profile['with_body']} ({profile['with_body'] * 100 // total}%)")
    print(f"  With labels        : {profile['with_labels']} ({profile['with_labels'] * 100 // total}%)")
    print(f"  Total comments     : {profile['total_comments']}")

    # Text length stats
    ts = profile["title_length_stats"]
    bs = profile["body_length_stats"]
    tw = profile["title_word_stats"]
    bw = profile["body_word_stats"]
    print(f"\n  Title length (chars): min={ts['min']}, max={ts['max']}, mean={ts['mean']}, median={ts['median']}")
    print(f"  Title length (words): min={tw['min']}, max={tw['max']}, mean={tw['mean']}, median={tw['median']}")
    print(f"  Body  length (chars): min={bs['min']}, max={bs['max']}, mean={bs['mean']}, median={bs['median']}")
    print(f"  Body  length (words): min={bw['min']}, max={bw['max']}, mean={bw['mean']}, median={bw['median']}")

    # Category distribution
    cat_dist = profile["category_distribution"]
    print(f"\n  HEURISTIC CATEGORY DISTRIBUTION:")
    max_count = max(cat_dist.values()) if cat_dist else 1
    for cat, count in cat_dist.items():
        pct = count * 100 // total
        bar = "█" * min(count * 35 // max_count, 35)
        print(f"    {cat:<12s} {count:>5d} ({pct:>3d}%)  {bar}")

    # Corpus keywords
    print(f"\n  TOP CORPUS KEYWORDS (TF-IDF):")
    for term, score in profile["corpus_keywords"]:
        print(f"    {term:<30s} {score:>8.4f}")

    # Per-category keywords
    for cat, kws in profile.get("category_keywords", {}).items():
        print(f"\n  TOP KEYWORDS — {cat.upper()}:")
        for term, score in kws[:10]:
            print(f"    {term:<30s} {score:>8.4f}")

    # Label distribution
    label_dist = profile.get("label_distribution", {})
    if label_dist:
        print(f"\n  GITHUB LABEL DISTRIBUTION:")
        for label, count in label_dist.items():
            print(f"    {label:<40s} {count:>5d}")
    else:
        print(f"\n  No GitHub labels in this dataset.")

    # Label vs heuristic comparison
    if label_dist:
        print(f"\n  NOTE: Heuristic categories are independent of GitHub labels.")
        print(f"  GitHub labels cover {profile['with_labels']}/{total} issues ({profile['with_labels'] * 100 // total}%).")
        print(f"  Heuristic categories cover {total}/{total} issues (100%).")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Save enriched dataset
# ---------------------------------------------------------------------------

def save_enriched(
    enriched_issues: list[dict[str, Any]],
    source_path: Path,
    output_dir: Path = ANALYZED_DIR,
) -> Path:
    """Save enriched issues (with categories + keywords) to data/analyzed/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem  # e.g. fastapi_fastapi_issues_20241106T...
    filepath = output_dir / f"{stem}_analyzed.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(enriched_issues, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d enriched issues to %s", len(enriched_issues), filepath)
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse an ingested GitHub issue dataset.",
    )
    parser.add_argument(
        "dataset", nargs="?", type=Path, default=None,
        help="Path to a JSON dataset file. If omitted, uses the latest in data/raw/.",
    )
    parser.add_argument(
        "--top-keywords", type=int, default=15,
        help="Number of top keywords to show (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save enriched dataset to data/analyzed/",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Resolve dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = find_latest_dataset()

    logger.info("Loading dataset: %s", dataset_path)
    issues = load_dataset(dataset_path)
    logger.info("Loaded %d issues", len(issues))

    # Profile
    profile = profile_corpus(issues, top_n_keywords=args.top_keywords)
    print_report(profile)

    # Save enriched
    if args.save:
        out = save_enriched(profile["enriched_issues"], dataset_path)
        print(f"Enriched dataset saved to: {out}")


if __name__ == "__main__":
    main()
