"""Sentiment scoring for lyrics with the two lexicon tools taught in class: VADER (main) and TextBlob (cross-check).

Why VADER is scored line by line: its compound score is a squashed sum of word valences, so on a whole song with
dozens of sentiment words it is pushed towards -1 or +1 and stops telling songs apart. A lyric line is close to the
short social-media sentence VADER was built for, so we score every line and summarise the lines per song.
"""
import re
import unicodedata

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from src import config

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_QUOTES = str.maketrans({"’": "'", "‘": "'"})

# Swear words and slurs that VADER's lexicon scores as negative, used for the profanity test in notebook 04.
# Only exact lexicon entries are listed, so removing them changes nothing else in VADER.
PROFANITY = [
    "apeshit", "ass", "assholes", "bastard", "bastards", "bitch", "bitches", "bitchy", "bullshit",
    "cocksucker", "cunt", "damn", "damned", "damnit", "dick", "dickhead", "dipshit", "fuck", "fucked",
    "fucker", "fuckers", "fucks", "goddam", "goddamn", "goddamned", "hell", "jackass", "motherfucker",
    "motherfucking", "niggas", "nigger", "shit", "shits", "shitty", "shithead", "slut", "sluts", "twat",
    "whore", "whores", "wtf",
]
PROFANITY_RE = re.compile(r"\b(" + "|".join(map(re.escape, PROFANITY)) + r")\b", flags=re.IGNORECASE)


def lyric_lines(text: str) -> list:
    """Split lyrics into non-empty lines after fixing invisible characters and curly apostrophes.

    Casing and punctuation are kept on purpose: VADER uses CAPS and '!' as intensity cues.
    """
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_QUOTES)
    return [line.strip() for line in text.split("\n") if line.strip()]


def vader_analyzer(drop_words=()) -> SentimentIntensityAnalyzer:
    """NLTK's VADER. Words in drop_words are removed from this analyzer's lexicon only (they then count as neutral)."""
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    for word in drop_words:
        analyzer.lexicon.pop(word, None)
    return analyzer


def score_lines(lines, analyzer, cache: dict) -> list:
    """VADER compound score per line. `cache` remembers lines already scored (choruses repeat a lot)."""
    scores = []
    for line in lines:
        if line not in cache:
            cache[line] = analyzer.polarity_scores(line)["compound"]
        scores.append(cache[line])
    return scores


def summarise_lines(scores) -> dict:
    """Song-level summary of its line scores: mean compound and the share of positive / negative lines."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return {"vader_mean": np.nan, "pos_share": np.nan, "neg_share": np.nan, "n_lines": 0}
    return {
        "vader_mean": s.mean(),
        "pos_share": (s >= config.VADER_POS).mean(),
        "neg_share": (s <= config.VADER_NEG).mean(),
        "n_lines": s.size,
    }


def vader_song_scores(texts: pd.Series, analyzer, cache: dict, unique_lines: bool = False) -> pd.DataFrame:
    """Line-level VADER summarised per song. unique_lines=True counts a repeated line (e.g. a chorus) only once.

    mean_line_chars is returned too: a very high value means the lyrics have no real line breaks, so "lines" are
    whole paragraphs and the per-line score saturates like a whole-song score would.
    """
    rows = []
    for text in texts:
        lines = lyric_lines(text)
        if unique_lines:
            lines = list(dict.fromkeys(lines))
        row = summarise_lines(score_lines(lines, analyzer, cache))
        row["mean_line_chars"] = np.mean([len(line) for line in lines]) if lines else 0.0
        rows.append(row)
    return pd.DataFrame(rows, index=texts.index)


def profane_line_share(texts: pd.Series) -> pd.Series:
    """Share of each song's lines that contain at least one word from PROFANITY."""
    def share(text):
        lines = lyric_lines(text)
        return np.mean([bool(PROFANITY_RE.search(line)) for line in lines]) if lines else np.nan
    return texts.map(share)


def textblob_song_scores(texts: pd.Series) -> pd.DataFrame:
    """TextBlob polarity (-1..1) and subjectivity (0..1) of the whole song.

    TextBlob averages the words it knows, so (unlike VADER's sum) it does not saturate on long texts.
    """
    rows = []
    for text in texts:
        sentiment = TextBlob("\n".join(lyric_lines(text))).sentiment
        rows.append({"tb_polarity": sentiment.polarity, "tb_subjectivity": sentiment.subjectivity})
    return pd.DataFrame(rows, index=texts.index)
