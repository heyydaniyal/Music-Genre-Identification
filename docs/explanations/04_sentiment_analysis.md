# Notebook 04 — Sentiment Analysis: explained from scratch

This note is for preparing the defence. It is **not** part of the report. It walks through notebook 04 in order and
explains every idea from zero. Numbers are the ones the notebook prints.

---

## 0. What the notebook is trying to do

The brief asks for "sentiment ratings" of lyrics and to compare them **within and across genres**. A sentiment rating is
a number that says how positive or negative a piece of text sounds. We compute one per song, then ask five questions:

| | Question |
|---|---|
| SQ1 | Which genre is most positive / negative, and how big are the differences? |
| SQ2 | How much of rap's negativity is just swear words? |
| SQ3 | Did sentiment change over the decades? |
| SQ4 | Are more positive songs more viewed? |
| SQ5 | Is the chorus more positive than the verses of the same song? |

---

## 1. The two sentiment tools

Both are **lexicon-based**: they have a dictionary of words with a score (a "valence") and combine the scores of the
words they find. Nothing is trained on our data, so neither is a "pretrained model" in the sense the brief forbids.

**VADER** (Hutto & Gilbert, 2014)
- Each word in its lexicon has a valence between about -4 and +4 (e.g. *love* +3.2, *hate* -2.7).
- Rules adjust them: a negation in the 3 words before ("don't love") flips and weakens the score; CAPS and "!!!" make it
  stronger; "but" gives more weight to what follows.
- It adds everything up and squashes the sum into [-1, 1] with `compound = s / sqrt(s² + 15)`.
- Usual cut-offs: compound ≥ 0.05 positive, ≤ -0.05 negative, in between neutral.

**TextBlob**
- Also a word list with polarity in [-1, 1], but it **averages** the polarities of the words it knows.
- Handles "not" only partly and ignores CAPS. Also gives *subjectivity* (0 = factual, 1 = opinion).

Worked example of saturation. Imagine a song with 40 positive words each worth +2: the VADER sum is s = 80,
compound = 80 / sqrt(6400 + 15) ≈ 0.999. Twenty words already give 40 / sqrt(1615) ≈ 0.995. So every long song ends up near
±1. That is why we score **lines** (few words each) and average the line scores.

**Defence questions**
- *Why VADER and not only TextBlob?* VADER handles negation ("I don't love you anymore": VADER -0.522, TextBlob +0.500)
  and intensity, which lyrics use a lot. Both were taught; TextBlob is our cross-check.
- *Isn't VADER made for tweets?* Yes, short informal text. A lyric line is short informal text, which is another reason to
  score per line.

---

## 2. Method checks (Section 2)

**2.1 Test lines.** We wrote 8 lines that mimic lyrics, each aimed at a weak spot. Results to remember: slang ("this beat
is sick", "we killed it") is read as negative by both tools; "Money on my mind" scores 0 because no word carries feeling.

**2.2 Scoring.** Per song we keep:
- `vader_mean`: mean compound of the song's lines (our main score);
- `pos_share`, `neg_share`: share of lines ≥ 0.05 and ≤ -0.05;
- `vader_mean_unique`: same mean, each distinct line counted once (to check if repeated choruses distort things);
- `tb_polarity`: TextBlob on the whole song (it averages, so no saturation).

A **cache** (a dictionary line → score) means each distinct line is scored only once; 1,150,890 distinct lines.

**2.3 Filter (D13).** Some songs have no line breaks, so a "line" is a whole paragraph and saturates. We keep songs with
≥ 5 lines and mean line length ≤ 150 characters. 618 of 39,663 songs removed (1.6%), rap most (2.5%).

**2.4 Whole-song check (commented code).** On 2,000 songs: whole-song VADER puts 66.8% of songs beyond ±0.9; the
line-level mean puts none there. Spearman between the two = 0.900, so the ranking is similar but the whole-song score
loses the strength of the feeling.

**2.5 Agreement.**
- *Spearman correlation*: rank both scores from lowest to highest and compute the ordinary correlation of the ranks.
  1 = same order, 0 = unrelated. VADER vs TextBlob = 0.563.
- *Cohen's kappa*: agreement on positive/neutral/negative corrected for chance:
  κ = (p_observed − p_chance) / (1 − p_chance). 0.294 = "fair".
  Small example: if two raters agree on 70% of songs, but by chance alone they'd agree 50% of the time,
  κ = (0.7 − 0.5)/(1 − 0.5) = 0.4.
- All lines vs unique lines: 0.943, so repetition doesn't drive the scores.
- Agreement in rap (0.551) is as high as elsewhere → no sign VADER is especially broken on rap.

**2.6 Extremes.** Reading the most extreme lines is the sentiment version of error analysis. It is how we found the
no-line-break songs (2.3).

**Defence questions**
- *How do you know VADER is right on lyrics?* We don't have human labels (limitation). We checked consistency (TextBlob
  agreement, repetition), saturation, known failure cases and the extremes.
- *Why 150 characters?* The median line is about 32 characters; 150 is almost 5× that, so only text without line breaks
  is caught.

---

## 3. SQ1 — genres (Section 3)

**Median and bootstrap CI.** The median is the middle song. The 95% bootstrap interval: draw as many songs as we have,
*with replacement* (some songs twice, some not at all), compute the median, repeat 1,000 times, keep the middle 95% of
those medians. It shows how much the median would move with a different sample.

Results: country 0.054, rb 0.050, pop 0.030, rock -0.003, rap -0.019.

**Kruskal-Wallis.** Puts all songs in one ranking (1 = most negative). If genre didn't matter, each genre's average rank
would be about the same. H measures how far apart the average ranks are. ε² = H/(n−1) = 0.079 → genre explains ~8% of the
variation.

**Mann-Whitney + Cliff's delta.** For two genres, compare every rap song with every country song: δ = P(rap higher) −
P(rap lower). Example: rap A = [1, 2], country B = [3]: rap is lower in both pairs → δ = 0 − 1 = −1.
Labels: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium. Rap vs country = -0.419 (medium).

**Holm.** With 10 pairs, some would look "significant" by luck. Holm sorts the p-values and multiplies the smallest by 10,
the next by 9, and so on.

**Defence questions**
- *Why medians and rank tests, not means and ANOVA?* Scores are bounded, skewed and piled up near 0; rank methods don't
  assume a normal distribution.
- *Everything is significant, so what?* With 10,000 songs per genre even tiny differences give tiny p-values. That's why
  we interpret effect sizes (δ, ε²).

---

## 4. SQ2 — profanity (Section 4)

We build a second VADER whose lexicon has swear words and slurs removed (list in `src/sentiment.py`, only words that
are in VADER's lexicon). Lines without those words keep their score, so only ~12% of rap lines change.

Results: rap median -0.019 → +0.025; negative rap songs 37.7% → 15.3%; rap vs pop δ -0.304 → -0.076. Rock becomes the
least positive genre.

**Defence question.** *Isn't removing those words cheating?* We report both versions. The point is to measure how much of
the gap is vocabulary: most of it.

---

## 5. SQ3 — decades (Section 5)

Songs from 1960 on, genre–decade cells with ≥ 100 songs. Spearman between year and `vader_mean` per genre, with bootstrap
interval (resample song pairs). Pop -0.008 and rap -0.012 (no trend); rock -0.081, country -0.097, rb -0.158 (slightly
more negative over time). Per decade, every genre except rap has a lower median now than in the 1960s; pop's drop is
all before 2000, and most pop songs are from after 2000, which is why its correlation is about 0.

**Defence question.** *Why not trust the 1960s numbers fully?* Few old songs, and they're the ones people still look up
(classics) — not a random sample of the decade.

---

## 6. SQ4 — popularity (Section 6)

Older songs had more time to collect views, and sentiment changes with year, so year could fake a relation. We rank each
song's views **within its genre and release year** (percentile 0–1, groups with ≥ 20 songs) and correlate that with
sentiment. All |ρ| < 0.1 → practically no relation. Correlation is not causation.

---

## 7. SQ5 — chorus vs verse (Section 7)

Paired comparison: each song against itself (chorus mean − verse mean). **Wilcoxon signed-rank**: rank the absolute
differences, add the ranks of the positive ones; if neither part were systematically higher this would be about half the
total. Result: chorus more positive by about +0.01 in every genre (52–54% of songs), not significant in country (p = 0.076).
Only songs with both tags (26% of pop, about half of rap/rb), so it describes the more structured songs.

---

## 8. Output

`results/sentiment_findings.csv`: one row per main number, read by notebook 05 so the conclusion quotes the same values.
