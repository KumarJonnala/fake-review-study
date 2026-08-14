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
DEFAULT_N_PER_CELL = 3                    # x16 cells = 48 reviews (sample size).
                                          # Raise to 10 for a full 160-review run.

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
# False if you switch to a model that respects the length instruction on its own.
# NOT yet re-measured on qwen2.5:32b — check the in-band rate on the first run and
# turn this off if the model hits the target unaided, since trimming is lossy.
DEFAULT_TRIM_OVERLONG = True

# Token ceiling per review = upper_band_chars / this. English runs ~4 chars/token,
# so /2 leaves ~2x headroom — the cap stops runaway output without truncating a
# compliant review. A single global cap let short reviews run to 470+ chars.
NUM_PREDICT_CHARS_PER_TOKEN = 2
NUM_PREDICT_FLOOR = 80

# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------
# `source` encodes polarity for the REAL half of the corpus (Ott et al. structure,
# confirmed by keyword rates: TripAdvisor 0.68 positive-word / 0.05 negative-word;
# Web 0.17 / 0.45). The MTurk (fake) half is mixed polarity and CANNOT be
# sentiment-matched — the original positive/negative split is absent from this CSV.

REAL_SOURCE_BY_SENTIMENT = {"positive": "TripAdvisor", "negative": "Web"}
FAKE_SOURCE = "MTurk"

REAL_LABEL = "real"
FAKE_LABEL = "fake"

# Length band examples are drawn from, per condition. Short cells get the shortest
# quartile; long cells get the interquartile middle.
EXAMPLE_QUANTILES = {
    "short": (0.0, 0.25),
    "long": (0.25, 0.75),
}

# Columns the examples CSV must contain.
REQUIRED_EXAMPLE_COLUMNS = {"Binary_label", "Category", "text", "source"}

# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

SENTIMENT_BRIEFS = {
    "positive": "an overall positive, satisfied experience",
    "negative": "an overall negative, disappointing experience",
}

# Aspects named in the `structured` condition.
ASPECTS = [
    "the location",
    "the price / value for money",
    "room cleanliness",
    "how long the stay was",
    "the staff",
    "noise levels",
    "breakfast",
]

# How many aspects to name per structured review (sampled from ASPECTS).
ASPECTS_PER_REVIEW = 4

PROMPT_OPENING = 'Write ONE realistic hotel guest review for a hotel called "{hotel}".'

PROMPT_SENTIMENT = "Sentiment: {brief}."

PROMPT_LENGTH_SHORT = (
    "Length: 1-2 sentences — about {chars} characters ({w_min}-{w_max} words). "
    "Keep it brief. Hard limits: no fewer than {lo} and no more than {hi} characters. "
    "Stop once you have made your point."
)

PROMPT_LENGTH_LONG = (
    "Length: a detailed review of about {chars} characters ({w_min}-{w_max} words) — "
    "a full, substantial review, not a summary. Hard limits: no fewer than {lo} and "
    "no more than {hi} characters. Do not pad beyond that."
)

PROMPT_STRUCTURED = (
    "Cover these specific aspects, woven naturally into prose "
    "(no bullet points, no headings): {aspects}."
)

PROMPT_UNSTRUCTURED = (
    "Write it as a free-flowing, natural review. Do not work through a checklist of "
    "aspects — let it read the way a real guest rambles about whatever stood out to them."
)

# Anti-templating nudge, applied to every cell.
PROMPT_DIVERSITY = (
    "Vary sentence structure, vocabulary, and concrete details (dates, room numbers, "
    "staff names, small incidents) so the output does not feel templated. Avoid filler "
    "phrases like 'overall a great experience' or 'would definitely recommend' unless a "
    "real reviewer would plausibly write them. Small imperfections help: casual tone, "
    "tangents, uneven pacing."
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
    "Example A — a genuine review written by a real hotel guest:\n{real_example}\n\n"
    "Example B — a review written by a paid writer imitating a guest:\n{fake_example}\n\n"
    "These show two different registers. Write a NEW review — do not copy phrasing "
    "from either. Follow the sentiment and length in your instructions, not the "
    "sentiment or length of the examples."
)

PROMPT_ZEROSHOT = "Write the review now."

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
    "source",
    # Factor metadata for the analysis.
    "length",
    "sentiment",
    "structure",
    "example_mode",
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

# LABEL LEAKAGE: drop `source` and `is_synthetic` before training. In the merged
# corpus both are perfect giveaways — source maps MTurk->fake, TripAdvisor/Web->real,
# <model id>->LLM, and is_synthetic is 1 on exactly the LLM class. Keep them as
# metadata for slicing results, never as model inputs.
LEAKY_COLUMNS = ["source", "is_synthetic"]
