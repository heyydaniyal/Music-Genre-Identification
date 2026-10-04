"""Factories for text representations (one place to define them, reused everywhere).

Planned API:
    make_vectorizer(kind, **params)   # 'bow' | 'tfidf_word' | 'tfidf_char'
    Word2VecVectorizer                # trained on our own training lyrics only (no pretrained resources)
    make_feature_union(spec)          # combines text, structural and metadata blocks via ColumnTransformer
"""
