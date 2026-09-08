# Offline Source Navigator

Offline Source Navigator is a small retrieval model package for searching a
trusted local document collection on ordinary hardware. Every returned chunk
includes its source path and content hash. A calibrated low score produces an
explicit `ABSTAIN_INSUFFICIENT_LOCAL_EVIDENCE` result.

The package trains two real classical retrieval models over an owned
Apache-2.0 corpus:

- `tfidf-v1`: learned unigram and bigram inverse-document-frequency weights;
- `lsa-v1`: the same TF-IDF representation reduced by fitted truncated SVD.

Both models use `szl_kernels.governed_cosine_topk`, which searches the document
matrix in bounded row blocks and writes the operation into the existing
`UnifiedReceiptChain`. BM25 evaluation uses the separately maintained
`szl-retrieval-bench` package.

## Reproduce locally

From the `szl-kernels` repository root:

```powershell
$python = "python"
& $python -m pip install ".[frontier]"
& $python -I -B .\frontier\offline_retrieval\train_retrieval.py
& $python -I -B .\frontier\offline_retrieval\evaluate_retrieval.py `
  --bench-root ..\szl-retrieval-bench
& $python -I -B .\frontier\offline_retrieval\navigator.py `
  --model lsa-v1 `
  --query "How does the receipt chain detect tampering?"
```

The build refuses missing sources, path escapes, duplicate query identifiers,
non-finite arrays, hash mismatches, or unsafe NumPy object arrays. It records
duplicate source bytes rather than silently training on them twice.

Use Python 3.11 or newer with the `frontier` extra installed. Local validation
used Python 3.11 and PyTorch 2.11.0; CUDA is optional. Queries outside the learned
vocabulary return a receipted abstention with `kernel_called=false`.

## Evidence boundary

The models are fitted locally and their retrieval metrics are measured on the
committed author-created evaluation queries. The judgments have no independent
adjudication, so the measurements establish reproducibility on this small
evaluation only. They do not establish general search quality, factual answer
quality, clinical fitness, regulatory approval, deployment, or benefit in a
population. The navigator retrieves cited source text; it does not diagnose,
prescribe, execute actions, access the network, or generate an answer beyond
the retrieved excerpts.

The intended next evaluation is a separately reviewed corpus in a concrete
community domain such as disaster preparedness, public-service navigation, or
offline education. Admission requires source licenses, relevance judgments,
and subject-matter review before any domain claim.
