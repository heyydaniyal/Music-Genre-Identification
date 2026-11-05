# Notebook 03 (Section 11) and Notebook 05 — explained from scratch

Defence notes, not part of the report.

---

## Notebook 03, Section 11: final model and Kaggle submission

**Which model is submitted?** The Section 7 configuration (word + character TF-IDF, balanced Linear SVM, C = 0.1)
**refitted on all 134,789 training songs**. Section 7 trained on 80% so the other 20% could measure performance; once
the settings are fixed that 20% has done its job, and using it gives the rare genres about 25% more examples. The CV score
(0.534 ± 0.004) and holdout score (0.541) describe this configuration. The refitted model agrees with the 80% model on
93.2% of the test songs.

**Memory.** The refit is the heaviest step: the word + character matrix of all songs is over 2 GB and LinearSVC
(liblinear) makes its own internal copy (~3 GB) while training. Peak ≈ 6 GB → run on a machine with ≥ 8 GB RAM,
16 GB comfortable. On a smaller machine, the fallback is the Section 7 model (trained on 80%).

**The pipeline, step by step**
1. `final_word_vectorizer.fit_transform(X)`: learns the word vocabulary from all training lyrics and turns each song into
   TF-IDF weights (unigrams + bigrams, `min_df=10` = words in fewer than 10 songs are ignored).
2. `final_char_vectorizer.fit_transform(X)`: the same with pieces of 3–5 characters inside word boundaries (`char_wb`).
   It catches spelling variants and slang ("lovin", "luv").
3. `hstack_csr`: glues the two matrices side by side → one row per song, 422,718 columns.
4. `LinearSVC(C=0.1, class_weight="balanced").fit`: one linear scorer per genre; the highest score wins (Annex A).
5. The test lyrics are only *transformed* with the fitted vectorizers, never fitted on → no test information leaks in.

**Why `hstack_csr` and the `del` lines exist.** scipy's `hstack` briefly needs about twice the result's size in extra
memory, which crashed the notebook. `hstack_csr` (in `src/features.py`) builds the same matrix block by block; a test
checks it is identical to scipy's result. The notebook also loads only the `id`, `clean` and `tag` columns and deletes
matrices once they are no longer needed. None of this changes any result.

**Sanity checks.** Without test labels we (1) compare the share of each predicted genre with the validation predictions,
and (2) check that the refitted model mostly agrees with the 80% model (93.2%). A broken pipeline would show up as a
strange distribution (e.g. everything pop) or low agreement.

**`write_submission`** refuses to write if: wrong number of rows, duplicated ids, ids different from Kaggle's sample file,
or an unknown genre. **SHA-256** is a fingerprint of the file's bytes: same hash = same file, byte for byte.

**Defence questions**
- *Why balanced class weights?* Macro F1 averages the six genres equally, so country (2.6% of songs) counts as much as
  pop (41%). Without weights the model mostly ignores country and rb.
- *Is the CV score optimistic?* C was chosen on 3-fold CV inside the training split, and the 5-fold CV uses all songs, so
  it is slightly optimistic; the holdout (0.541) and CV (0.534) agree, which is reassuring. Kaggle's public score is an
  independent check.

---

## Notebook 05

No new analysis: it loads `results/experiment_log.csv`, `results/submission_log.csv` and `results/sentiment_findings.csv`
so every quoted number comes from notebooks 03 and 04.

**Annexes** contain only methods not taught in the course: linear SVM with class weights, Kruskal–Wallis + ε²,
Mann–Whitney + Cliff's delta, Holm, Wilcoxon signed-rank, bootstrap + Spearman, Cohen's kappa. Read them once before
the defence; each is explained in plain words with its formula. The worked examples are in
`docs/explanations/04_sentiment_analysis.md`.

**Likely questions on the conclusion**
- *Which hypothesis failed?* Country is not easy to recognise (F1 ≈ 0.35) even though its vocabulary is distinctive: few
  country songs actually use that vocabulary, and there are few country songs to learn from.
- *Main limitation?* No human-labelled sentiment data, so we know VADER is consistent and where it is biased, not how
  accurate it is on lyrics.
- *What would you do next?* Human labels for a few hundred lines; a lyrics-adapted lexicon for slang and swearing; an
  emotion lexicon; rhyme/structure features for the classifier.
