# Decision log

One row per decision. IDs match the project roadmap (§9.3).

| ID | Date | Decision | Evidence / reason |
|---|---|---|---|
| D1 | 2026-10-04, changed 2026-11 | ~~Metadata allowed, Track M final~~ → **Lyrics only.** Artist, year, views, features and title are not used as model features | The brief asks to predict genre "from its lyrics"; decision by Dani in Nov 2026 |
| D2 | 2026-10-04 | One notebook per phase (01–05) plus a shared `src/` package; notebook markdown is the report | Answers to Q4, Q7 |
| D3 | 2026-11 | Validation: stratified 80/20 holdout (seed 42) for model selection; stratified 5-fold CV for the final configuration | Notebook 03, Sections 2 and 7 |
| D4 | 2026-10-04 | No transformers, no pretrained embeddings or models | Answer to Q6 ("no pretrained") |
| D5 | 2026-11 | Cleaning frozen as in notebook 02 (lowercase, section tags as tokens, stopwords kept); final representation word (1,2) + char_wb (3,5) TF-IDF, balanced LinearSVC C=0.1 | Notebook 03: holdout macro F1 0.541, CV 0.534 ± 0.004; stopword and section-token checks in Section 5 |
| D6 | 2026-11-04 | Sentiment questions: SQ1 genre, SQ2 profanity, SQ3 decades, SQ4 popularity (within year), SQ5 chorus vs verse | Brief questions (a)–(c) + hypotheses in notebook 01 |
| D8 | 2026-10-04 | Deliver by Sat 12 Dec (latest Sun 13 Dec 17:59) for the full +1 early bonus | Answer to Q2 |
| D9 | 2026-10-04 | Claude drafts and explains the code; Dani reviews, runs and defends every cell | Answer to Q10 |
| D10 | 2026-11-04 | misc excluded from the sentiment analysis | Notebook 01: misc is mostly literature, scripts and tables, not music |
| D11 | 2026-11-04 | VADER (main, scored per line) + TextBlob (cross-check). No other lexicons | Both taught in the sentiment practical class (slides "07_Sentiment_Analysis") |
| D12 | 2026-11-05 | Sentiment on **all** songs of the five music genres, no sampling (an earlier 10,000-per-genre sample was dropped) | Dani: runtime is not a constraint; results then describe the whole training set |
| D13 | 2026-11-04 | Drop songs with < 5 lines or mean line length > 150 characters from the sentiment analysis | Reading the extremes showed lyrics without line breaks saturating at ±1 |
| D14 | 2026-11-04 | No human-labelled sentiment validation; method checked with test lines, whole-song comparison, TextBlob agreement and reading extremes | Dani's decision; stated as the main limitation in notebook 05 |
| D15 | 2026-11-05 | Final Kaggle model refit on 100% of the training songs (same settings as notebook 03 Section 7) | Dani: memory is not a constraint; needs ~6 GB RAM at peak, so run on ≥ 8 GB (16 GB comfortable) |
