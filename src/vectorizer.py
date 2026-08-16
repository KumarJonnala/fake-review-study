from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(cfg):
    return TfidfVectorizer(
        lowercase=cfg.get("lowercase", True),
        strip_accents=cfg.get("strip_accents", "unicode"),
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.95),
        max_features=cfg.get("max_features", 100000),
        sublinear_tf=True,
    )
