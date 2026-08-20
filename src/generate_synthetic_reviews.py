"""
Synthetic Hotel Review Generator — 16-cell factorial
=====================================================

Generates LLM-written hotel reviews crossing four binary axes:

    length      : short | long
    sentiment   : positive | negative
    structure   : structured (aspect-hinted) | unstructured (free-flow)
    example_mode: few_shot (1 real + 1 fake example) | zero_shot (pure instruction)

2 x 2 x 2 x 2 = 16 cells, --n-per-cell reviews each.

All tunable parameters — paths, axes, length targets, prompt wording, retry behaviour,
CLI defaults — live in `config.py`. Edit that file to change the standing configuration;
use CLI flags for one-off runs. This module holds only logic.

Intended use: building and stress-testing fake-review DETECTION models, and measuring
which generation conditions produce reviews that evade a detector. NOT intended for
posting on real review platforms to mislead consumers — that violates most platforms'
terms of service and consumer-protection law (e.g. FTC rules on fake reviews) in many
jurisdictions.


LABEL LEAKAGE — READ BEFORE TRAINING ANYTHING ON THIS OUTPUT
------------------------------------------------------------
Drop `model`, `is_synthetic`, and (once merged with the human corpus) `source` from
your feature set. All are perfect giveaways:

    model:         non-null on exactly the LLM class, NaN everywhere else
    is_synthetic:  1 on exactly the LLM class, 0 everywhere else
    source:        set only on human rows after a merge (MTurk -> fake,
                    TripAdvisor/Web -> real); NaN on every LLM row

A classifier handed either column scores ~100% and has learned nothing about language.
Keep them as metadata for slicing results, never as inputs.


Requires a running Ollama server with the target model pulled:
    ollama serve
    ollama pull qwen2.5:32b

Usage:
    python3 src/generate_synthetic_reviews.py --dry-run
    python3 src/generate_synthetic_reviews.py --model llama3.2:3b --n-per-cell 1
    python3 src/generate_synthetic_reviews.py --model qwen2.5:32b --n-per-cell 10
"""

import argparse
import csv
import itertools
import json
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import config as cfg

# The full factorial: 2 x 2 x 2 x 2 = 16 cells. Built once, since the hotel rotation
# is keyed off a cell's position in this list.
ALL_CELLS = list(
    itertools.product(cfg.LENGTHS, cfg.SENTIMENTS, cfg.STRUCTURES, cfg.EXAMPLE_MODES)
)


# ---------------------------------------------------------------------------
# Example pool
# ---------------------------------------------------------------------------


class ExamplePool:
    """Draws few-shot examples from the human corpus (read-only)."""

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        missing = cfg.REQUIRED_EXAMPLE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")

        df = df.dropna(subset=["text"]).copy()
        df["n_chars"] = df["text"].str.len()
        self.df = df
        self.hotels = sorted(df["Category"].dropna().unique())
        if not self.hotels:
            raise ValueError(f"No hotel categories found in {csv_path}")

    def _pick(self, pool: pd.DataFrame, hotel: str, length: str, rng: random.Random) -> str:
        """Prefer the same hotel; prefer a length band suiting the cell."""
        if pool.empty:
            return ""

        same_hotel = pool[pool["Category"] == hotel]
        candidates = same_hotel if not same_hotel.empty else pool

        # Short cells get the shortest quartile; long cells get the middle band.
        q_lo, q_hi = cfg.EXAMPLE_QUANTILES[length]
        lo = candidates["n_chars"].quantile(q_lo)
        hi = candidates["n_chars"].quantile(q_hi)
        banded = candidates[(candidates["n_chars"] >= lo) & (candidates["n_chars"] <= hi)]

        if banded.empty:
            banded = candidates

        idx = rng.choice(list(banded.index))
        return str(self.df.at[idx, "text"]).strip()

    def get_pair(self, sentiment: str, hotel: str, length: str, rng: random.Random):
        """Return (real_example, fake_example).

        The real example is sentiment-matched via `source`. The fake example is drawn
        from MTurk and CANNOT be sentiment-matched — polarity is absent from this CSV.
        """
        real_source = cfg.REAL_SOURCE_BY_SENTIMENT[sentiment]
        real_pool = self.df[
            (self.df["Binary_label"] == cfg.REAL_LABEL)
            & (self.df["source"] == real_source)
        ]
        fake_pool = self.df[
            (self.df["Binary_label"] == cfg.FAKE_LABEL)
            & (self.df["source"] == cfg.FAKE_SOURCE)
        ]

        return (
            self._pick(real_pool, hotel, length, rng),
            self._pick(fake_pool, hotel, length, rng),
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def length_band(target_chars: int, tolerance: float):
    """Acceptable character range around a target. Default +/-35% of long (700)
    gives 455-945, which sits inside the real corpus IQR of 487-988."""
    return int(target_chars * (1 - tolerance)), int(target_chars * (1 + tolerance))


def build_system_prompt(length, sentiment, structure, hotel, targets, rng, tolerance):
    """Returns (system_prompt, opener_move).

    The opener move is returned so it can be written to the CSV as a factor column:
    the analysis needs to know how each review was told to open, and the validator
    replays this same function to recover it.
    """
    target = targets[length]
    w_min, w_max = target["words"]

    lo, hi = length_band(target["chars"], tolerance)
    length_template = (
        cfg.PROMPT_LENGTH_SHORT if length == "short" else cfg.PROMPT_LENGTH_LONG
    )

    parts = [
        cfg.PROMPT_OPENING.format(hotel=hotel),
        cfg.PROMPT_SENTIMENT.format(brief=cfg.SENTIMENT_BRIEFS[sentiment]),
        length_template.format(
            chars=target["chars"], w_min=w_min, w_max=w_max, lo=lo, hi=hi
        ),
    ]

    if structure == "structured":
        chosen = rng.sample(cfg.ASPECTS, k=min(cfg.ASPECTS_PER_REVIEW, len(cfg.ASPECTS)))
        parts.append(cfg.PROMPT_STRUCTURED.format(aspects=", ".join(chosen)))
    else:
        parts.append(cfg.PROMPT_UNSTRUCTURED)

    # Drawn after the structure clause so the RNG sequence stays deterministic per cell.
    moves, weights = zip(*cfg.OPENER_MOVES)
    opener_move = rng.choices(moves, weights=weights, k=1)[0]
    parts.append(cfg.PROMPT_OPENER.format(move=opener_move))

    parts.append(cfg.PROMPT_DIVERSITY)
    parts.append(cfg.PROMPT_OUTPUT_RULE)

    return "\n".join(parts), opener_move


def build_messages(length, sentiment, structure, example_mode, hotel, pool, targets, rng,
                   tolerance):
    """Returns (messages, opener_move)."""
    system_prompt, opener_move = build_system_prompt(
        length, sentiment, structure, hotel, targets, rng, tolerance
    )
    messages = [{"role": "system", "content": system_prompt}]

    if example_mode == "few_shot":
        real_ex, fake_ex = pool.get_pair(sentiment, hotel, length, rng)
        if real_ex and fake_ex:
            content = cfg.PROMPT_FEWSHOT.format(real_example=real_ex, fake_example=fake_ex)
        else:
            # No usable pair in the corpus — fall back rather than send a half-built prompt.
            content = cfg.PROMPT_ZEROSHOT
    else:
        content = cfg.PROMPT_ZEROSHOT

    messages.append({"role": "user", "content": content})
    return messages, opener_move


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*|<thought>.*?</thought>\s*", re.S | re.I)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure[,!]?|certainly[,!]?|of course[,!]?|here(?:'s| is)[^:\n]*|review)\s*[:\-—]\s*",
    re.I,
)


def normalize_ascii(text: str) -> str:
    """Fold typographic Unicode to the ASCII the human corpus uses.

    The corpus is 100% ASCII across 1592 reviews; qwen2.5:32b put a non-ASCII character
    in 46.9% of a 160-review run, which alone separates the classes at 100% precision.
    That is an encoding asymmetry between the two halves of the dataset, not a style
    difference, so it is corrected here rather than left for a classifier to exploit.

    Explicit map first (so em dash becomes "--" rather than being stripped), then NFKD to
    reduce accented Latin to its base letter. Characters with no ASCII equivalent -- CJK
    above all -- are left in place and rejected by `non_latin_chars` in generate_one, since
    silently deleting them would leave a mangled half-sentence in the dataset.
    """
    for src, dst in cfg.ASCII_NORMALIZATION.items():
        text = text.replace(src, dst)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def clean_response(text: str, normalize: bool = None) -> str:
    text = _THOUGHT_RE.sub("", text or "")
    if cfg.DEFAULT_NORMALIZE_ASCII if normalize is None else normalize:
        text = normalize_ascii(text)
    text = text.strip()

    # Strip a leading "Sure, here's a review:" style preamble (possibly on its own line).
    prev = None
    while prev != text:
        prev = text
        text = _PREAMBLE_RE.sub("", text).strip()

    # Strip symmetric wrapping quotes.
    for opener, closer in (('"', '"'), ("'", "'"), ("“", "”")):
        if len(text) > 1 and text.startswith(opener) and text.endswith(closer):
            text = text[1:-1].strip()

    # Collapse runaway blank lines but keep paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------


def call_ollama(url, model, messages, temperature, num_predict):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    resp = requests.post(url, json=payload, timeout=cfg.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def trim_to_sentence(text: str, max_chars: int) -> str:
    """Cut back to the last sentence boundary that fits. Never cuts mid-word."""
    if len(text) <= max_chars:
        return text
    window = text[: max_chars + 1]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut == -1:
        for end in (".", "!", "?"):
            cut = max(cut, window.rfind(end))
    if cut <= 0:
        return text  # no sentence boundary to cut at — leave it, don't mangle it
    return text[: cut + 1].strip()


def non_latin_chars(text: str) -> set:
    """Letters outside Latin (incl. Latin-1/Extended), plus CJK punctuation.

    qwen2.5 is Chinese-developed and code-switches: a 160-review run produced
    "the staff, especially前台的简，非常乐于助人。" mid-sentence. The human corpus is
    100% Latin script, so a single CJK character is a perfect label giveaway -- and
    it is also just broken data. Treated as a degenerate output and retried, the same
    way output-too-short is, rather than being written to the CSV.

    Accented Latin ("café", "Zürich") is legitimate in English hotel reviews and passes.
    """
    stray = set()
    for c in str(text):
        if c.isalpha() and ord(c) > 0x024F:      # beyond Latin Extended-B
            stray.add(c)
        elif c in "，。、！？；：（）「」『』【】":  # full-width CJK punctuation
            stray.add(c)
    return stray


def generate_one(url, model, messages, temperature, num_predict, target_chars,
                 tolerance, length_attempts, trim_overlong):
    """Generate one review, preferring output inside the target length band.

    Returns (text, n_attempts, None) on success or (None, n_attempts, error) on failure.
    Length misses do not fail the row — we keep the closest attempt and record its
    length, so the analysis can filter. Only API/degenerate failures fail the row.
    """
    lo, hi = length_band(target_chars, tolerance)
    last_error = "unknown error"
    best = None  # (distance_from_band, text)
    attempts = 0

    for _ in range(max(1, length_attempts)):
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            attempts += 1
            try:
                raw = call_ollama(url, model, messages, temperature, num_predict)
                text = clean_response(raw)
                if len(text.split()) < cfg.MIN_ACCEPTABLE_WORDS:
                    last_error = f"output too short ({len(text.split())} words)"
                    tqdm.write(f"  [WARN] attempt {attempt}: {last_error}, retrying")
                    time.sleep(cfg.RETRY_SLEEP)
                    continue

                stray = non_latin_chars(text)
                if stray:
                    last_error = f"non-English output ({''.join(sorted(stray))[:20]})"
                    tqdm.write(f"  [WARN] attempt {attempt}: {last_error}, retrying")
                    time.sleep(cfg.RETRY_SLEEP)
                    continue

                n = len(text)
                distance = 0 if lo <= n <= hi else min(abs(n - lo), abs(n - hi))
                if best is None or distance < best[0]:
                    best = (distance, text)
                if distance == 0:
                    return text, attempts, None
                break  # got usable text, just the wrong length — try a fresh sample
            except Exception as exc:  # noqa: BLE001 — network/JSON/key errors retry alike
                last_error = f"{type(exc).__name__}: {exc}"
                tqdm.write(f"  [WARN] attempt {attempt} failed: {last_error}")
                time.sleep(cfg.RETRY_SLEEP)

    if best is None:
        return None, attempts, last_error

    text = best[1]
    if trim_overlong and len(text) > hi:
        text = trim_to_sentence(text, hi)
    return text, attempts, None


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_completed(output_path: Path):
    """Map cell_id -> count of rows already written."""
    if not output_path.exists():
        return {}
    try:
        done = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return {}
    if "cell_id" not in done.columns:
        return {}
    return done["cell_id"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate synthetic hotel reviews across a 16-cell factorial design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default=cfg.DEFAULT_MODEL, help="Any model pulled in Ollama")
    p.add_argument("--n-per-cell", type=int, default=cfg.DEFAULT_N_PER_CELL,
                   help=f"Reviews per cell (x{len(ALL_CELLS)} cells). Ignored if "
                        "--total-reviews is given.")
    p.add_argument("--total-reviews", type=int, default=None,
                   help="Total reviews for this run, split as evenly as possible across "
                        f"all {len(ALL_CELLS)} cells (overrides --n-per-cell). Hotel "
                        "rotation offsets are always computed from the full factorial, "
                        "so a --cells filter still reproduces the same per-cell counts.")
    p.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", cfg.DEFAULT_HOST),
        help="Ollama host:port",
    )
    p.add_argument("--examples-csv", type=Path, default=cfg.DEFAULT_EXAMPLES_CSV,
                   help="Human corpus, read-only: few-shot examples and hotel names")
    p.add_argument("--output", type=Path, default=cfg.DEFAULT_OUTPUT, help="Output CSV (created)")

    p.add_argument("--short-chars", type=int, default=cfg.LENGTH_TARGETS["short"]["chars"])
    p.add_argument("--short-words-min", type=int, default=cfg.LENGTH_TARGETS["short"]["words"][0])
    p.add_argument("--short-words-max", type=int, default=cfg.LENGTH_TARGETS["short"]["words"][1])
    p.add_argument("--long-chars", type=int, default=cfg.LENGTH_TARGETS["long"]["chars"],
                   help="Default matches the real corpus median (700 chars)")
    p.add_argument("--long-words-min", type=int, default=cfg.LENGTH_TARGETS["long"]["words"][0])
    p.add_argument("--long-words-max", type=int, default=cfg.LENGTH_TARGETS["long"]["words"][1])

    p.add_argument("--length-tolerance", type=float, default=cfg.DEFAULT_LENGTH_TOLERANCE,
                   help="Accepted deviation from the char target (0.35 -> long = 455-945, "
                        "inside the real corpus IQR of 487-988)")
    p.add_argument("--length-attempts", type=int, default=cfg.DEFAULT_LENGTH_ATTEMPTS,
                   help="Fresh samples to draw trying to land in the band; closest is kept")
    p.add_argument("--trim-overlong", action="store_true",
                   default=cfg.DEFAULT_TRIM_OVERLONG,
                   help="Trim still-overlong output back to a sentence boundary")

    p.add_argument("--temperature", type=float, default=cfg.DEFAULT_TEMPERATURE,
                   help="Higher = more diverse")
    p.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED,
                   help="Seeds hotel and example sampling")
    p.add_argument("--cells", default=None,
                   help="Substring/glob filter on cell_id, e.g. 'long_positive*'")
    p.add_argument("--dry-run", action="store_true", help="Print prompts, make no API calls")
    p.add_argument("--resume", action="store_true", help="Skip cells already in --output")

    return p.parse_args(argv)


def distribute_total(total: int, n_cells: int):
    """Split `total` as evenly as possible across `n_cells` buckets.

    200 / 16 = 12.5: the first `remainder` cells get one
    extra rather than leaving reviews unassigned or overshooting the total.
    """
    base, remainder = divmod(total, n_cells)
    return [base + 1 if i < remainder else base for i in range(n_cells)]


def cell_matches(cell_id: str, pattern: str) -> bool:
    if not pattern:
        return True
    from fnmatch import fnmatch

    return fnmatch(cell_id, pattern) or pattern in cell_id


def main(argv=None):
    args = parse_args(argv)

    targets = {
        "short": {
            "chars": args.short_chars,
            "words": (args.short_words_min, args.short_words_max),
        },
        "long": {
            "chars": args.long_chars,
            "words": (args.long_words_min, args.long_words_max),
        },
    }

    # Per-condition token ceiling. English runs ~4 chars/token; allow ~2x headroom over
    # the upper band so the cap only stops runaway output, never a compliant review.
    # A single global cap let short reviews run to 470+ chars in testing.
    num_predict_by_length = {
        name: max(
            cfg.NUM_PREDICT_FLOOR,
            int(
                length_band(t["chars"], args.length_tolerance)[1]
                / cfg.NUM_PREDICT_CHARS_PER_TOKEN
            ),
        )
        for name, t in targets.items()
    }

    url = f"http://{args.host}{cfg.OLLAMA_CHAT_PATH}"
    pool = ExamplePool(args.examples_csv)

    # Per-cell review counts and hotel-rotation offsets are both computed from the FULL
    # cell list (not the filtered `combos`) so a --cells filter or --resume never shifts
    # which hotel a given (cell, rep) gets, and a given cell_id always gets the same
    # target count regardless of what else is being run alongside it.
    if args.total_reviews is not None:
        counts = distribute_total(args.total_reviews, len(ALL_CELLS))
    else:
        counts = [args.n_per_cell] * len(ALL_CELLS)
    cell_target = dict(zip(ALL_CELLS, counts))

    cell_offset = {}
    running = 0
    for c in ALL_CELLS:
        cell_offset[c] = running
        running += cell_target[c]

    combos = [c for c in ALL_CELLS if cell_matches("_".join(c), args.cells)]
    if not combos:
        print(f"No cells match --cells {args.cells!r}", file=sys.stderr)
        return 1

    total_reviews = sum(cell_target[c] for c in combos)
    print(f"Ollama:     {url}")
    print(f"Model:      {args.model}")
    print(f"Cells:      {len(combos)} cells, {total_reviews} reviews total")
    print(f"Lengths:    short ~{args.short_chars} chars, long ~{args.long_chars} chars")
    print(f"Examples:   {args.examples_csv} ({len(pool.hotels)} hotels, read-only)")
    print(f"Output:     {args.output}")
    print()

    completed = load_completed(args.output) if args.resume else {}

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.output.exists() or args.output.stat().st_size == 0
        out_file = args.output.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_file, fieldnames=cfg.CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            out_file.flush()
        failures_path = args.output.with_name(args.output.stem + "_failures.jsonl")
        fail_file = failures_path.open("a", encoding="utf-8")
    else:
        out_file = writer = fail_file = failures_path = None

    n_ok = n_failed = n_skipped = 0

    try:
        for length, sentiment, structure, example_mode in combos:
            cell_id = f"{length}_{sentiment}_{structure}_{example_mode}"
            combo = (length, sentiment, structure, example_mode)
            n_target = cell_target[combo]
            already = completed.get(cell_id, 0)
            if already >= n_target:
                print(f"[skip] {cell_id} — {already} rows already present")
                n_skipped += n_target
                continue

            # Seed per cell so a resumed or filtered run reproduces the same choices.
            rng = random.Random(f"{args.seed}:{cell_id}")

            # Burn the RNG draws belonging to rows already written. Without this a
            # resumed run gives rep `already` the draws that belong to rep 0, so its
            # prompts differ from an uninterrupted run and validate_generated.py's
            # replay (which walks rep 0..n) no longer matches what was sent.
            for skipped in range(already):
                ordinal = cell_offset[combo] + skipped
                build_messages(
                    length, sentiment, structure, example_mode,
                    pool.hotels[ordinal % len(pool.hotels)], pool, targets, rng,
                    args.length_tolerance,
                )

            todo = range(already, n_target)
            desc = f"{cell_id:<45}"
            for rep in tqdm(todo, desc=desc, unit="rev", leave=True):
                # Round-robin over ALL 20 hotels, continuing across cells rather than
                # restarting each one. Indexing by `rep` alone would pin every cell to
                # the same first few hotels (and to hotels[0] entirely at n-per-cell=1),
                # reintroducing the constant-hotel confound this design exists to avoid.
                ordinal = cell_offset[combo] + rep
                hotel = pool.hotels[ordinal % len(pool.hotels)]
                messages, opener_move = build_messages(
                    length, sentiment, structure, example_mode, hotel, pool, targets, rng,
                    args.length_tolerance,
                )

                if args.dry_run:
                    print("=" * 78)
                    print(f"CELL {cell_id}  rep={rep}  hotel={hotel}")
                    print("=" * 78)
                    for m in messages:
                        print(f"--- {m['role']} ---")
                        print(m["content"])
                    print()
                    break  # one sample prompt per cell is enough to inspect

                text, n_attempts, error = generate_one(
                    url,
                    args.model,
                    messages,
                    args.temperature,
                    num_predict_by_length[length],
                    targets[length]["chars"],
                    args.length_tolerance,
                    args.length_attempts,
                    args.trim_overlong,
                )

                if error is not None:
                    n_failed += 1
                    # Never write a placeholder review into the dataset — a failed
                    # generation must not become a training example.
                    fail_file.write(
                        json.dumps(
                            {
                                "cell_id": cell_id,
                                "rep_index": rep,
                                "hotel": hotel,
                                "model": args.model,
                                "error": error,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        + "\n"
                    )
                    fail_file.flush()
                    continue

                writer.writerow(
                    {
                        "Binary_label": cfg.OUTPUT_BINARY_LABEL,
                        "Category": hotel,
                        "domain": cfg.OUTPUT_DOMAIN,
                        "text": text,
                        "is_synthetic": cfg.OUTPUT_IS_SYNTHETIC,
                        "model": args.model,
                        "length": length,
                        "sentiment": sentiment,
                        "structure": structure,
                        "example_mode": example_mode,
                        "opener_move": opener_move,
                        "cell_id": cell_id,
                        "rep_index": rep,
                        "n_chars": len(text),
                        "n_words": len(text.split()),
                        "target_chars": targets[length]["chars"],
                        "in_length_band": int(
                            length_band(targets[length]["chars"], args.length_tolerance)[0]
                            <= len(text)
                            <= length_band(targets[length]["chars"], args.length_tolerance)[1]
                        ),
                        "n_attempts": n_attempts,
                        "gen_temperature": args.temperature,
                        "gen_seed": args.seed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                out_file.flush()  # incremental write: survive interruption
                n_ok += 1
    finally:
        if out_file:
            out_file.close()
        if fail_file:
            fail_file.close()

    print()
    print("=" * 60)
    if args.dry_run:
        print(f"Dry run — {len(combos)} cells, no API calls made.")
    else:
        print(f"Generated: {n_ok}")
        print(f"Failed:    {n_failed}" + (f"  -> {failures_path}" if n_failed else ""))
        if n_skipped:
            print(f"Skipped:   {n_skipped} (already present, --resume)")
        print(f"Output:    {args.output}")

        # Achieved vs target length — the `long` condition is meant to sit inside the
        # real corpus band, so surface it rather than making the user go looking.
        if n_ok and args.output.exists():
            done = pd.read_csv(args.output)
            print()
            print("Length (target -> achieved):")
            for name in cfg.LENGTHS:
                rows = done[done["length"] == name]
                if rows.empty:
                    continue
                lo, hi = length_band(targets[name]["chars"], args.length_tolerance)
                hit = rows["in_length_band"].mean() * 100
                print(
                    f"  {name:<5} target {targets[name]['chars']:>4} chars "
                    f"(band {lo}-{hi})  ->  mean {rows['n_chars'].mean():>6.0f}, "
                    f"in band {hit:.0f}%"
                )
            if done["in_length_band"].mean() < 0.6:
                print("  NOTE: poor length adherence — small models are weak at this.")
                print("        Try a larger model, --length-attempts 3, or --trim-overlong.")
        print()
        leaky = " and ".join(f"`{c}`" for c in cfg.LEAKY_COLUMNS)
        print(f"Reminder: drop {leaky} before training — both leak the label.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
