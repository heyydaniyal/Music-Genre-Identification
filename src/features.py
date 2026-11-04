"""Helpers for the feature matrices of notebook 03."""
import numpy as np
from scipy import sparse


def hstack_csr(left, right, chunk_rows: int = 20_000):
    """Put two CSR matrices side by side, like scipy.sparse.hstack, using much less memory.

    scipy's hstack first concatenates all the data of both inputs and then builds the result, so it briefly needs about
    twice the size of the result in extra memory. With the word + character TF-IDF matrices (about 1.8 GB together) that
    was enough to crash an 8 GB laptop. Here the result arrays are allocated once and filled a block of rows at a time,
    so the only extra memory is one small block. The result is identical to scipy.sparse.hstack(..., format="csr").
    """
    if left.shape[0] != right.shape[0]:
        raise ValueError("Both matrices must have the same number of rows.")
    left, right = sparse.csr_matrix(left), sparse.csr_matrix(right)

    indptr = left.indptr.astype(np.int64) + right.indptr.astype(np.int64)
    n_cols = left.shape[1] + right.shape[1]
    index_dtype = np.int32 if max(indptr[-1], n_cols) < np.iinfo(np.int32).max else np.int64
    data = np.empty(indptr[-1], dtype=np.result_type(left.dtype, right.dtype))
    indices = np.empty(indptr[-1], dtype=index_dtype)

    for start in range(0, left.shape[0], chunk_rows):
        stop = min(start + chunk_rows, left.shape[0])
        block = sparse.hstack([left[start:stop], right[start:stop]], format="csr")
        data[indptr[start]:indptr[stop]] = block.data
        indices[indptr[start]:indptr[stop]] = block.indices

    return sparse.csr_matrix((data, indices, indptr.astype(index_dtype)), shape=(left.shape[0], n_cols))
