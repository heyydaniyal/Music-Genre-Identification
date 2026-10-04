"""Text cleaning, driven by a single config dict so ablations toggle options instead of copying code.

Planned API:
    DEFAULT_CLEANING: dict                     # the frozen configuration (set at Gate G4)
    normalize_unicode(text)                    # NFKC, zero-width and odd whitespace removal
    handle_section_tags(text, mode)            # 'remove' | 'keep' | 'extract'
    clean_text(text, **cfg)                    # applies the steps above in a fixed order
    clean_series(series, cfg)                  # vectorized wrapper used by every notebook
    split_sections(text)                       # {'chorus': ..., 'verse': ...} for the whole-vs-sections test
"""
