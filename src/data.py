"""Loading raw data and passing data between notebooks.

Planned API:
    load_raw(split)                    -> DataFrame       # 'train' | 'test', utf-8, dtypes fixed
    save_processed(df, name, config)   -> Path            # writes data/processed/<name>.pkl + config hash
    load_processed(name, config)       -> DataFrame       # asserts the file exists and the hash matches (no stale handoffs)
    config_hash(config: dict)          -> str
"""
