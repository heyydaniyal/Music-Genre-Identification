"""Sentiment scoring for lyrics.

Planned API:
    vader_line_scores(text)          # VADER per line (avoids compound-score saturation on long texts)
    aggregate_song(line_scores)      # mean compound, share of positive/negative lines, variance
    score_corpus(series, method)     # 'vader' (+ others per decision D11)
"""
