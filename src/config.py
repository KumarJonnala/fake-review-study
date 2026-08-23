"""
Configuration for the synthetic hotel review generator.

Everything tunable lives here: paths, the factorial axes, length targets, prompt
wording, retry behaviour, and the CLI defaults. `generate_synthetic_reviews.py`
holds only logic.

Values here are DEFAULTS. Anything also exposed as a CLI flag can be overridden
per-run without editing this file — edit here to change the standing configuration
for the study, use flags for one-off experiments.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The corpus CSVs are READ-ONLY inputs. Output goes to its own file, and the
# generator never writes to, appends to, or overwrites the source data.

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

DEFAULT_EXAMPLES_CSV = DATA_DIR / "Hotel_Human_VS_HumanFake_relabelled.csv"

# Model-tagged, so runs with different LLMs never mix in one file (and a later run
# with a different model is not silently skipped by --resume).
DEFAULT_OUTPUT = DATA_DIR / "generated" / "Hotel_LLM_Reviews_qwen2.5_32b.csv"

# ---------------------------------------------------------------------------
# Model / API defaults  (CLI: --model, --host, --temperature, --seed)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen2.5:32b"             # the study model, run on the cluster GPU.
                                          # ~20GB pull, needs >=24GB VRAM to stay on-GPU.
                                          # Locally, override: --model llama3.2:latest
DEFAULT_HOST = "localhost:11434"          # overridden by the OLLAMA_HOST env var
OLLAMA_CHAT_PATH = "/api/chat"            # /api/chat, not /api/generate: few-shot
                                          # needs a real message list
DEFAULT_TEMPERATURE = 1.0                 # high, for lexical diversity across samples
DEFAULT_SEED = 12                         # seeds hotel rotation + example sampling
DEFAULT_N_PER_CELL = 10                   # x16 cells = 160 reviews.
                                          # Matches N_PER_CELL in slurm_synthetic_data.sh:
                                          # keep the two in step, or a local run and a
                                          # cluster run silently produce different volumes.

# The four models the study compares. slurm_synthetic_data.sh loops over this list by
# default (override with MODEL=<tag> for a single one-off run).
MODELS = [
    "gemma4:e4b",
    "ministral-3:8b",
    "llama3.2:3b",
    "qwen3.5:9b",
]

# Total reviews generated per model, spread across all 16 cells.
DEFAULT_TOTAL_REVIEWS_PER_MODEL = 200

# ---------------------------------------------------------------------------
# Factorial axes  —  2 x 2 x 2 x 2 = 16 cells
# ---------------------------------------------------------------------------

LENGTHS = ["short", "long"]
SENTIMENTS = ["positive", "negative"]
STRUCTURES = ["structured", "unstructured"]
EXAMPLE_MODES = ["few_shot", "zero_shot"]

# ---------------------------------------------------------------------------
# Length targets  (CLI: --short-chars, --long-chars, --{short,long}-words-{min,max})
# ---------------------------------------------------------------------------
# Fixed constants — NOT recomputed from the CSV at run time.
#
# The `long` values come from a one-time measurement of the real half of
# data/Hotel_Human_VS_HumanFake_relabelled.csv (Binary_label == "real", n=796):
#     median 700 chars / 125 words, IQR 487-988 chars
# so `long` reviews land inside the real corpus's own length distribution.
#
# `short` is deliberately below the real minimum (151 chars / 25 words) to make it a
# genuine contrast. Consequence for analysis: length alone separates short-cell
# reviews from real ones — report per-cell, and consider a length-matched control.

LENGTH_TARGETS = {
    "short": {"chars": 200, "words": (20, 35)},
    "long": {"chars": 700, "words": (110, 140)},
}

# Accepted deviation from the char target. 0.35 puts `long` at 455-945 chars,
# inside the real corpus IQR of 487-988.
DEFAULT_LENGTH_TOLERANCE = 0.35

# Fresh samples to draw trying to land in the band; the closest is kept.
# Small models are poor at length control: on a 48-review llama3.2 run, 39 of 48
# needed the second sample. Raise to 3 if in-band rates stay low.
DEFAULT_LENGTH_ATTEMPTS = 2

# Trim still-overlong output back to a sentence boundary. ON, because llama3.2
# overshoots the length target badly without it (long reviews averaged 1097 chars
# against a 700 target, outside the real corpus IQR). Note this is a real edit to
# the generated text — it can cut a review before its closing sentiment. Set to
# False if you switch to a model that respects the length instruction on its own,
# or pass --no-trim-overlong for a single run without editing this file.
# NOT yet re-measured on qwen2.5:32b — check the in-band rate on the first run and
# turn this off if the model hits the target unaided, since trimming is lossy.
# Whichever way it ends up, the value is recorded per run in the _prompts.jsonl
# sidecar, so no output is ever ambiguous about whether its text was trimmed.
DEFAULT_TRIM_OVERLONG = True

# Token ceiling per review = upper_band_chars / this. English runs ~4 chars/token,
# so /2 leaves ~2x headroom — the cap stops runaway output without truncating a
# compliant review. A single global cap let short reviews run to 470+ chars.
NUM_PREDICT_CHARS_PER_TOKEN = 2
NUM_PREDICT_FLOOR = 80

# ---------------------------------------------------------------------------
# Few-shot examples  —  ONE FIXED PAIR, shared by every prompt and every model
# ---------------------------------------------------------------------------
# Pinned by corpus row index, not drawn at run time. The examples used to be resampled
# per review (sentiment-matched by `source`, length-banded by cell, same-hotel preferred),
# which meant `few_shot` was not one condition but ~100 different ones, and no two models
# ever saw the same examples. Fixing the pair makes few_shot vs zero_shot a clean
# contrast, and makes cross-model comparison exact.
#
# Constants rather than a seeded draw: a draw would still shift if the seed, the pool
# filters or the RNG call order ever changed. These indices cannot.
#
# WHAT THIS GIVES UP, deliberately:
#   - Sentiment matching. `source` encodes polarity for the REAL half (TripAdvisor 0.68
#     positive-word / 0.05 negative-word; Web 0.17 / 0.45), so a per-sentiment draw used
#     to hand positive cells a positive example. One pair cannot. PROMPT_FEWSHOT already
#     covers this: it tells the model to follow the sentiment and length in its
#     instructions, not the example's -- wording that existed because the MTurk half is
#     mixed polarity and never could be matched.
#   - Length banding. Short cells no longer get a short example. Both texts below sit at
#     the real corpus median (~710 / ~680 chars) so neither cell type gets an outlier.
#
# WHAT TO WATCH: with 100 few-shot reviews per model seeing the SAME two texts, any
# borrowed phrasing is now borrowed 100 times instead of once. validate_generated.py
# section 4 measures verbatim 6-gram copying and currently reports 0; if that rises, the
# examples are being parroted and this decision needs revisiting.
#
# Both chosen near the median length of their half, and both POSITIVE, so the pair
# contrasts register -- genuine guest vs paid writer -- rather than polarity, which is
# what PROMPT_FEWSHOT says the two examples are there to show.
#   real: row 3    TripAdvisor, omni,  707 chars
#   fake: row 647  MTurk,       james, 681 chars
# Indices are into DEFAULT_EXAMPLES_CSV, which is a read-only input; ExamplePool asserts
# the label and source on load, so a wrong index fails loudly rather than silently
# feeding the wrong register into every prompt.
FEWSHOT_REAL_INDEX = 3
FEWSHOT_FAKE_INDEX = 647

FAKE_SOURCE = "MTurk"
REAL_SOURCE = "TripAdvisor"

REAL_LABEL = "real"
FAKE_LABEL = "fake"

# Columns the examples CSV must contain.
REQUIRED_EXAMPLE_COLUMNS = {"Binary_label", "Category", "text", "source"}

# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------
# PUNCTUATION RULE: no em-dashes, en-dashes, or DOUBLE QUOTES inside any PROMPT_* string.
# The model mirrors the punctuation register of its own instructions. An earlier version
# of this file used em-dashes freely, and the llama3.2 output came back with en-dashes in
# 14.6% of reviews against 0.0% in BOTH human classes (n=1592) -- a perfect label giveaway
# caused entirely by the prompt file. Spaced hyphens ran 64.6% vs 18.3% in real reviews
# for the same reason.
#
# Double quotes are the same trap: the human corpus contains ZERO double quotes of any
# kind (straight or curly) across all 1592 reviews, so any generated review using one is
# instantly identifiable. PROMPT_OPENING used to wrap the hotel name in them.
#
# Numeric ranges (1-2 sentences, {w_min}-{w_max}) and single quotes are fine; apostrophes
# appear in ~65% of human reviews. Comments and docstrings are never sent to the model
# and are exempt. src/validate_generated.py section 7 scans for the next one of these
# generically, so it gets caught without anyone knowing to look for it.

SENTIMENT_BRIEFS = {
    "positive": "an overall positive, satisfied experience",
    "negative": "an overall negative, disappointing experience",
}

# Aspects named in the `structured` condition.
# Phrased as JUDGEMENTS to make, not slots to fill.
#
# The earlier wording named topics as bare nouns -- "how long the stay was", "breakfast" --
# and models answered each with one stock value. On a gemma4 run, 59% of stay-length
# mentions were "four nights" (the real corpus spreads over 18 values, top share 21%), and
# every single breakfast mention was "pastries" (the corpus uses 13 distinct items, top
# share 28%). The rate of mentioning each topic was human-like; only the VALUE collapsed.
#
# A slot has one obvious filler. A judgement has many phrasings, so the same instruction
# admits far more surface variety without weakening the structured-vs-unstructured
# contrast, which is a factorial axis and must stay.
#
# Twelve, not the original seven. With ASPECTS_PER_REVIEW fixed at 4, a longer list means
# any given aspect is named in ~33% of structured reviews instead of ~57%, so the same
# instruction produces a wider spread of subject matter across a 200-review run.
#
# The five additions are grounded in what the real corpus actually discusses, not invented:
# bathroom 30.7% of real reviews, bed/sleep 29.0%, decor 28.5%, the view 19.3%, wifi 13.7%.
# Check-in scored highest of all the candidates at 41.8% but is deliberately absent -- the
# staff regex already matches it, so it would double-count rather than add an aspect.
#
# REORDERING OR RESIZING THIS LIST SHIFTS THE RNG. `rng.sample(ASPECTS, 4)` draws
# positions, so 4-of-12 is a different stream from 4-of-7 and CSVs generated under the
# old list no longer replay. Rewording an entry in place is safe; changing the length is
# not, and was done here deliberately.
#
# Every entry must also be a key of ASPECT_PATTERNS in validate_generated.py, which looks
# aspects up by exact string -- a missing key would score zero coverage silently, so there
# is an import-time guard there that raises instead.
ASPECTS = [
    "how convenient the location was",
    "whether it felt worth the price",
    "how clean the room was",
    "whether the stay felt too short or too long",
    "how the staff behaved",
    "how quiet or noisy it was",
    "what breakfast was like",
    "what the bathroom was like",
    "how well you slept",
    "whether the decor felt dated or fresh",
    "what the view was like",
    "whether the wifi worked",
]

# How many aspects to name per structured review (sampled from ASPECTS).
ASPECTS_PER_REVIEW = 4

PROMPT_OPENING = "Write ONE realistic hotel guest review for the hotel {hotel}."

PROMPT_SENTIMENT = "Sentiment: {brief}."

PROMPT_LENGTH_SHORT = (
    "Length: 1-2 sentences, about {chars} characters ({w_min}-{w_max} words). "
    "Keep it brief. Hard limits: no fewer than {lo} and no more than {hi} characters. "
    "Stop once you have made your point."
)

PROMPT_LENGTH_LONG = (
    "Length: a detailed review of about {chars} characters ({w_min}-{w_max} words). "
    "Write a full, substantial review, not a summary. Hard limits: no fewer than {lo} and "
    "no more than {hi} characters. Do not pad beyond that."
)

PROMPT_STRUCTURED = (
    "Cover these specific aspects, woven naturally into prose "
    "(no bullet points, no headings): {aspects}."
)

PROMPT_UNSTRUCTURED = (
    "Write it as a free-flowing, natural review. "
    "Let it read the way a real guest rambles about whatever stood out to them."
)

# WHY THERE IS NO OPENER INSTRUCTION
# ----------------------------------
# There used to be one: an OPENER_MOVES pool of 7 weighted categories, one sampled per
# review and injected as "Open the review with {move}." It was added because a generic
# "vary your openings" line did nothing -- llama3.2 collapsed to 6 distinct openers across
# 48 reviews, long cells opening "I" 100% of the time.
#
# It was removed because it became the ceiling rather than the floor. Grouping a 183-review
# qwen3.5 run by the move each was given showed the collapse happening INSIDE a mandated
# category: the 40 reviews told "first person plural, starting with the word We or Our"
# gave 15 distinct two-word openings, led by "we arrived" (6), "our three" (5), "we loved"
# (5). Two of the seven categories named the literal first word and carried 46% of the
# draw weight, which contradicted the pool's own stated design of steering a CATEGORY
# rather than a word. The distribution went with it:
#
#     opener   human real   steered
#     my            9.4%       0.5%   <- singular category named only "I"
#     this          7.3%       0.0%   <- no category covered it
#     the           8.4%      19.1%
#
# and 17 of the 25 commonest human openers (this, after, while, when, if, booked, first,
# great, ...) never appeared at all. Distinct opening words: 32 in 183, against a human
# 95% range of 44-61 at the same n. The real corpus gets its spread from a long tail --
# 109 words used three times or fewer, covering 19.1% of reviews -- which seven fixed
# categories cannot reproduce by construction.
#
# So this is a deliberate re-test of the 6-in-48 result, which was measured before the
# detail-spec removal above. If opener diversity does not improve without steering, that
# finding stands and the answer is a WIDER pool, not no pool -- do not simply restore the
# old 7 categories, which are now known to suppress "my" and "this".

# Anti-templating nudge, applied to every cell.
#
# DO NOT NAME SPECIFIC DETAIL TYPES HERE. This line used to ask for "concrete details
# (dates, room numbers, staff names, small incidents)", and because that wording is
# identical in all 200 prompts, every model converged on the same handful of concretes.
# Room or floor numbers appeared in 44-57.5% of generated reviews against 5.8% of the
# human corpus -- llama3.2 wrote "room on the Nth floor" 15 times in 200. The
# anti-templating instruction was itself the template, and at 10x the human rate it was
# also a detection tell in its own right.
#
# Same failure mode as the PUNCTUATION RULE above: prompt wording manufacturing an
# artifact that then looks like an LLM signature. Keep the pressure toward concreteness
# generic; naming the concretes is what does the damage.
PROMPT_DIVERSITY = (
    "The reviews should look like genuine user opinions."
    "Vary sentence structure, vocabulary, and concrete details so the output does not "
    "feel templated."
)

PROMPT_OUTPUT_RULE = (
    "Output ONLY the review text. No preamble, no surrounding quotation marks, "
    "no title, no label."
)

# Few-shot framing. The two examples are deliberately of DIFFERENT types — one
# genuine, one human-written fake — to show the model both registers. The closing
# instruction matters: the MTurk example cannot be sentiment-matched, so the model
# must be told to follow the instructions rather than the example's tone.
PROMPT_FEWSHOT = (
    "Here are two existing reviews of the same kind of hotel.\n\n"
    "Example A, a genuine review written by a real hotel guest:\n{real_example}\n\n"
    "Example B, a review written by a paid writer imitating a guest:\n{fake_example}\n\n"
    "These show two different registers. Write a NEW review. Do not copy phrasing "
    "from either. Follow the sentiment and length in your instructions, not the "
    "sentiment or length of the examples."
)

PROMPT_ZEROSHOT = "Write the review now."

# ---------------------------------------------------------------------------
# ASCII normalisation
# ---------------------------------------------------------------------------
# The human corpus contains ZERO non-ASCII characters across all 1592 reviews -- it was
# ASCII-normalised when it was built. qwen2.5:32b emits ordinary typographic Unicode, so
# 46.9% of a 160-review run carried at least one non-ASCII character. That makes
# `any(ord(c) > 127)` a classifier with 47% recall at 100% precision, before any model is
# trained.
#
# This is a PREPROCESSING ASYMMETRY, not an LLM style tell: it exists because the two
# halves of the dataset were encoded differently. Applying the corpus's own normalisation
# to the generated text makes the comparison valid. It is not the same as laundering
# stylistic tells -- semicolons (48.1% vs 6.6%), sentence length, and word choice are all
# left untouched, because those are real signal.
#
# Targets are chosen to land in character classes the corpus actually uses:
#   em dash -> "--"  (corpus uses "--" in 5.7% of reviews)
#   en dash -> "-"   (corpus 40.5%)
#   curly apostrophe -> "'" (corpus 63.3%)
#   double quote -> "'"  the corpus has NO double quotes of any kind, so converting to a
#                        single quote keeps the quoting function in a class it does use
# Accented Latin (cafe/Zurich) is stripped to its base letter via NFKD.
# CJK cannot be normalised and is rejected outright at generation time instead.

ASCII_NORMALIZATION = {
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‒": "-",    # figure dash
    "―": "--",   # horizontal bar
    "‘": "'",    # left single quote
    "’": "'",    # right single quote / typographic apostrophe
    "‚": "'",
    "“": "'",    # left double quote  -> single: corpus has no double quotes
    "”": "'",    # right double quote
    "„": "'",
    '"': "'",    # straight double quote, likewise absent from the corpus
    "…": "...",  # ellipsis (corpus uses "..." in 11.2%)
    " ": " ",    # non-breaking space
    " ": " ",
    " ": " ",
    "′": "'",
    "″": "'",
    "´": "'",
    "`": "'",
}

# Turn off to keep raw model output and normalise downstream in the classifier pipeline
# instead. Off means the generated CSV will not be encoding-comparable to the corpus.
DEFAULT_NORMALIZE_ASCII = True

# ---------------------------------------------------------------------------
# Retry / validation
# ---------------------------------------------------------------------------

MAX_RETRIES = 3          # API attempts per sample before giving up
RETRY_SLEEP = 2          # seconds between attempts
REQUEST_TIMEOUT = 300    # seconds per HTTP request
MIN_ACCEPTABLE_WORDS = 10  # shorter than this is degenerate output — retry

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
# Constant field values on every generated row.

OUTPUT_BINARY_LABEL = "fake"
OUTPUT_DOMAIN = "Hotel"
OUTPUT_IS_SYNTHETIC = 1

CSV_COLUMNS = [
    # Six corpus columns first, so pd.concat with the human data just works.
    "Binary_label",
    "Category",
    "domain",
    "text",
    "is_synthetic",
    "model",
    # Factor metadata for the analysis.
    "length",
    "sentiment",
    "structure",
    "example_mode",
    # No `opener_move`: opener steering was removed (see the block above PROMPT_DIVERSITY).
    # Its absence is also the provenance marker -- a CSV carrying that column was generated
    # WITH steering, and its RNG stream differs from anything produced since.
    "cell_id",
    "rep_index",
    "n_chars",
    "n_words",
    "target_chars",
    "in_length_band",
    "n_attempts",
    "gen_temperature",
    "gen_seed",
    "timestamp",
]

# LABEL LEAKAGE: drop `model` and `is_synthetic` before training
# once merged with the human corpus, which uses that column for MTurk/TripAdvisor/Web.
# All are perfect giveaways — model is non-null on exactly the LLM class,
# is_synthetic is 1 on exactly the LLM class. Keep them as metadata for slicing
# results, never as model inputs.
LEAKY_COLUMNS = ["model", "is_synthetic"]
