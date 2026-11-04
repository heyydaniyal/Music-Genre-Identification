# Notebook 03 (Section 11) and Notebook 05 — explained from scratch

Defence notes, not part of the report.

---

## Notebook 03, Section 11: final model and Kaggle submission

**Which model is submitted?** Exactly the one evaluated in Section 7: word + character TF-IDF with a balanced Linear SVM
(C = 0.1), trained on the 80% split. Its scores: holdout macro F1 0.541, 5-fold CV 0.534 ± 0.004.

**Why not refit on 100% of the songs?** That is the usual last step (25% more examples for the rare genres). We tried it:
memory reached 5.2 GB just after building the character vocabulary on all songs, and the process was killed on a 6 GB
machine. A notebook that crashes on the graders' laptops is "severely penalized", so the refit stays as a commented cell
with the reason, and we submit the model whose scores we measured. Defence answer: "we traded a small expected gain for
reproducibility; the submitted model is exactly the evaluated one".

**The pipeline, step by step**
1. `tfidf_bigram_vectorizer` (already fitted on the training split): turns each test song into word TF-IDF weights
   (unigrams + bigrams, `min_df=10` = words seen in fewer than 10 training songs are ignored).
2. `char_tfidf_vectorizer` (already fitted): the same with pieces of 3–5 characters inside word boundaries (`char_wb`).
   It catches spelling variants and slang ("lovin", "luv").
3. `hstack_csr`: glues the two matrices side by side → one row per song, ~365,000 columns.
4. `linear_svm_combined_tuned.predict`: one linear scorer per genre; the highest score wins (Annex A).
5. Vectorizers are only *applied* to the test set (`transform`), never fitted on it → no test information leaks in.

**Memory (why `hstack_csr` and the `del` lines exist).** The character matrix is ~1.5 GB. scipy's `hstack` briefly
needs about twice the result's size in extra memory, which crashed the notebook on a 6 GB limit (= an 8 GB laptop with
the OS running). `hstack_csr` (in `src/features.py`) builds the same matrix block by block; a test checks it is identical
to scipy's result. The notebook also loads only the `id`, `clean` and `tag` columns and deletes matrices once they are no
longer needed. None of this changes any result.

**Sanity check.** Without test labels we compare the share of each predicted genre with the training shares and the
validation predictions. A broken pipeline would show up as a strange distribution (e.g. everything pop).

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
