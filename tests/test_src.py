"""Checks for the src package. Run from the repo root with:  python -m pytest -q"""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import config, data, evaluation, sentiment, stats, submission


# ---------------- stats ----------------
def test_cliffs_delta_matches_brute_force():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40), rng.integers(1, 6, 30)       # integers -> many ties
    brute = np.mean([np.sign(x - y) for x, y in itertools.product(a, b)])
    assert stats.cliffs_delta(a, b) == pytest.approx(brute)


def test_cliffs_delta_extremes():
    assert stats.cliffs_delta([5, 6, 7], [1, 2]) == pytest.approx(1.0)
    assert stats.cliffs_delta([1, 2], [5, 6, 7]) == pytest.approx(-1.0)
    assert stats.delta_size(0.1) == "negligible" and stats.delta_size(-0.5) == "large"


def test_holm_known_values():
    # sorted p = .01, .02, .04 with m=3 -> 0.03, 0.04, 0.04 (monotone)
    assert stats.holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert stats.holm([0.9, 0.8]).max() <= 1.0


def test_kruskal_epsilon2_range():
    rng = np.random.default_rng(1)
    same = stats.kruskal_epsilon2({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    diff = stats.kruskal_epsilon2({"a": rng.normal(size=500), "b": rng.normal(3, size=500)})
    assert 0 <= same["epsilon2"] < 0.02 < diff["epsilon2"] <= 1


def test_bootstrap_and_spearman_ci():
    x = np.arange(200)
    low, high = stats.bootstrap_ci(x, np.median, n_boot=200)
    assert low <= np.median(x) <= high
    r = stats.spearman_ci(x, x * 2 + 1, n_boot=50)
    assert r["rho"] == pytest.approx(1.0) and r["ci_low"] == pytest.approx(1.0)


# ---------------- sentiment ----------------
def test_lyric_lines_cleans_invisible_characters():
    text = "I don’t ​care\n\n  \nYeah yeah  \n"
    assert sentiment.lyric_lines(text) == ["I don't care", "Yeah yeah"]
    assert sentiment.lyric_lines("") == []


def test_dropping_words_only_affects_that_analyzer():
    full = sentiment.vader_analyzer()
    clean = sentiment.vader_analyzer(drop_words=["shit"])
    assert full.polarity_scores("shit")["compound"] < 0
    assert clean.polarity_scores("shit")["compound"] == 0
    assert full.polarity_scores("shit")["compound"] < 0          # first analyzer unchanged


def test_vader_song_scores_and_cache():
    analyzer, cache = sentiment.vader_analyzer(), {}
    texts = pd.Series(["I love you\nI love you\nI hate this", "", "la la la"], index=[10, 11, 12])
    out = sentiment.vader_song_scores(texts, analyzer, cache)
    assert list(out.index) == [10, 11, 12]
    assert out.loc[10, "n_lines"] == 3 and out.loc[10, "pos_share"] == pytest.approx(2 / 3)
    assert np.isnan(out.loc[11, "vader_mean"]) and out.loc[11, "n_lines"] == 0
    assert out.loc[12, "vader_mean"] == 0 and out.loc[12, "mean_line_chars"] == 8
    assert out.loc[11, "mean_line_chars"] == 0
    uniq = sentiment.vader_song_scores(texts, analyzer, cache, unique_lines=True)
    assert uniq.loc[10, "n_lines"] == 2
    assert set(cache) == {"I love you", "I hate this", "la la la"}


def test_profanity_regex_whole_words_only():
    assert sentiment.PROFANITY_RE.search("What the HELL")
    assert not sentiment.PROFANITY_RE.search("hello shell class")


def test_profane_line_share():
    out = sentiment.profane_line_share(pd.Series(["oh shit\nhello\nno way\nDamn!", ""]))
    assert out.iloc[0] == pytest.approx(0.5) and np.isnan(out.iloc[1])


def test_textblob_scores_shape():
    out = sentiment.textblob_song_scores(pd.Series(["The food was great!", ""]))
    assert out.loc[0, "tb_polarity"] > 0 and out.loc[1, "tb_polarity"] == 0


# ---------------- data / logging / submission (written to a temporary folder) ----------------
@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(config, "SUBMISSIONS_DIR", tmp_path)
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": [3, 1, 2], "tag": ["pop", "pop", "pop"]}).to_csv(sample, index=False)
    monkeypatch.setattr(config, "SAMPLE_SUBMISSION_CSV", sample)
    sub_log = tmp_path / "submission_log.csv"
    sub_log.write_text("version,file_name,date,exp_id,sha256,cv_macro_f1,public_lb_macro_f1,notes\n")
    monkeypatch.setattr(config, "SUBMISSION_LOG", sub_log)
    exp_log = tmp_path / "experiment_log.csv"
    exp_log.write_text("exp_id,date,notebook,cleaning_config,representation,model,params,"
                       "cv_macro_f1_mean,cv_macro_f1_std,cv_weighted_f1,runtime_s,notes\n")
    monkeypatch.setattr(config, "EXPERIMENT_LOG", exp_log)
    return tmp_path


def test_load_processed_missing_file_explains(tmp_paths):
    with pytest.raises(FileNotFoundError, match="02_preprocessing"):
        data.load_processed("does_not_exist")


def test_write_submission_valid_and_ordered(tmp_paths):
    path = submission.write_submission([1, 2, 3], ["rap", "rock", "pop"], version=7)
    assert path.name == f"{config.GROUP_ID}_Version07.csv"
    out = pd.read_csv(path)
    assert list(out["id"]) == [3, 1, 2] and list(out["tag"]) == ["pop", "rap", "rock"]
    assert b"\r" not in path.read_bytes()          # same bytes, so same hash, on every OS


@pytest.mark.parametrize("ids, preds", [
    ([1, 2], ["pop", "pop"]),                    # wrong row count
    ([1, 2, 4], ["pop", "pop", "pop"]),          # wrong id set
    ([1, 1, 2], ["pop", "pop", "pop"]),          # duplicated id
    ([1, 2, 3], ["pop", "jazz", "pop"]),         # unknown label
])
def test_write_submission_rejects_bad_input(tmp_paths, ids, preds):
    with pytest.raises(ValueError):
        submission.write_submission(ids, preds, version=1)


def test_log_submission_keeps_one_row_and_lb_score(tmp_paths):
    path = submission.write_submission([1, 2, 3], ["rap", "rock", "pop"], version=2)
    log = submission.log_submission(path, 0.5)
    log.loc[0, "public_lb_macro_f1"] = 0.51
    log.to_csv(config.SUBMISSION_LOG, index=False)
    log = submission.log_submission(path, 0.52)
    assert len(log) == 1 and log.loc[0, "public_lb_macro_f1"] == 0.51 and log.loc[0, "cv_macro_f1"] == 0.52
    assert log.loc[0, "sha256"] == submission.file_sha256(path)


def test_log_experiment_replaces_row(tmp_paths):
    evaluation.log_experiment("E1", notebook="03", cv_macro_f1_mean=0.5)
    log = evaluation.log_experiment("E1", notebook="03", cv_macro_f1_mean=0.6)
    assert len(log) == 1 and log.loc[0, "cv_macro_f1_mean"] == 0.6
    with pytest.raises(ValueError):
        evaluation.log_experiment("E2", not_a_column=1)


# ---------------- features ----------------
def test_hstack_csr_matches_scipy():
    from scipy import sparse
    from src.features import hstack_csr
    a = sparse.random(53, 7, density=0.3, format="csr", random_state=1)
    a = sparse.csr_matrix(a.toarray() * (np.arange(53) != 5)[:, None])   # row 5 empty
    b = sparse.random(53, 11, density=0.2, format="csr", random_state=2)
    expected = sparse.hstack([a, b], format="csr")
    got = hstack_csr(a, b, chunk_rows=10)
    assert got.shape == expected.shape
    assert (got != expected).nnz == 0
    assert np.array_equal(got.indptr, expected.indptr) and np.array_equal(got.indices, expected.indices)
    with pytest.raises(ValueError):
        hstack_csr(a, b[:10])
