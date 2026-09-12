# Measured local results

Refreshed on 2026-09-12 UTC using Python 3.11.9, NumPy 2.4.6,
scikit-learn 1.9.0, and PyTorch 2.11.0+cu128. Training, evaluation, and the query
checks below ran locally on CPU. The models were fitted over 113 chunks from
11 admitted sources; no duplicate source files were excluded. TF-IDF has
7,973 features and LSA has 112 dimensions.

This document is excluded from the explicitly enumerated corpus manifest.
It reports the run; adding it does not change the model's admitted source bytes.

## Retrieval and abstention

Calibration uses 8 author-created queries. Evaluation uses a separate 12
author-created queries: 8 answerable and 4 unanswerable. The splits are disjoint
by identifier and normalized query text, but their relevance judgments have
not been independently adjudicated.

Ranking metrics use unique source paths, taking the first retrieved chunk for
each source. They are averaged over the 8 answerable queries **before**
abstention. They do not measure generated answers.

| Method | nDCG@5 | Recall@5 | P@5 | R-precision | MRR | MAP |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.754925 | 0.812500 | 0.350000 | 0.708338 | 0.750000 | 0.769444 |
| LSA | 0.754925 | 0.812500 | 0.350000 | 0.708338 | 0.750000 | 0.769444 |
| BM25 | 0.772586 | 0.875000 | 0.375000 | 0.645838 | 0.770833 | 0.738194 |

R-precision values preserve the benchmark implementation's emitted rounding.
BM25 uses `szl-retrieval-bench` revision
`280a94c204ce67ffbb995d99bd2e92a917f913d2`. Exact benchmark module hashes are
recorded in `artifacts/evaluation_report.json`.

| Method | Answerable accepted | Answerable abstained | Unanswerable abstained | Unanswerable accepted | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 6/8 | 2/8 | 4/4 | 0/4 | 0.875000 |
| LSA | 8/8 | 0/8 | 2/4 | 2/4 | 0.750000 |
| BM25 | 8/8 | 0/8 | 4/4 | 0/4 | 1.000000 |

"Accepted" means the score passed the calibrated threshold; it does not mean
every returned chunk was relevant. On this small set BM25 has higher nDCG@5,
Recall@5, and balanced abstention accuracy than either fitted model. TF-IDF and
LSA have higher MAP. These results do not establish general superiority for any
method. LSA accepts all eight answerable queries but also accepts two of four
unanswerable queries. This is a measured weakness, not a publishable quality
result; its publication hold remains in force.

The evaluation status is `HOLD_EXTERNAL_EVALUATION_REQUIRED`.
The model and report explicitly record `publication_eligible=false`,
`evaluation_independently_adjudicated=false`, and
`general_retrieval_quality=false`. Receipts are `UNSIGNED_HONEST`.
No acceleration, clinical-validity, production-runtime, or general
out-of-distribution detection claim is made.

## Reproducibility

These commands ran successfully from the repository root:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
python -I -B .\frontier\offline_retrieval\train_retrieval.py
python -I -B .\frontier\offline_retrieval\train_retrieval.py
python -I -B .\frontier\offline_retrieval\evaluate_retrieval.py --bench-root ..\szl-retrieval-bench
```

The two consecutive training runs produced identical SHA256 values for all
seven training artifacts. This verifies reproducibility in the recorded local
environment with the thread limits shown above; this does not establish
byte-identical results across different BLAS thread counts. The source-byte
admission hashes were refreshed after removing the kernel's NumPy byte bridge
and integrating protected source `ea507322d8e2ce90ce6e78f6889ab1c8f4874b0b`.
That merged Lambda validation change also changes indexed API documentation.
The corpus still has 113 chunks, but its text and fitted artifacts differ from
the September 7/8 runs. The table above reports the new run, including the LSA
unanswerable-query failures; historical scores must not be attributed to it.
Scoped `.gitattributes` rules retain LF source and metadata bytes in checkouts.
NumPy archives retain their existing Git LFS binary attributes.

| Artifact | SHA256, identical in both runs |
|---|---|
| `documents.json` | `83815da4ba345da0e910d9adb584b43958df10253c17cb8be8786eb3bd4b65a1` |
| `lsa-components.npz` | `f6a7c4e5fb65ed391ff285bffe845a7346fc376f6c5ddadbae4e28e3c3559a6d` |
| `lsa-documents.npz` | `87cb85fc14e5286e985f9fee713f594aad41ce25e3fb07f03846d35addeb899f` |
| `model_manifest.json` | `3fe886de224dda7af02d80d96b0a130034a99514d99d434de6de060acfb51c21` |
| `tfidf-documents.npz` | `a07576ebeffce515f19390201ca7a9c4cae85e828e853b296745a0242f566b72` |
| `training_receipt.json` | `ffc59513c2ffebdbaabf36a1c404300058a217bd9ce7544a785452095ef36be4` |
| `vectorizer.json` | `4c3510801136a21e49381f0780bca70910e1ea89975329c33ef9eef0afe3e70c` |

The shared training receipt chain head is
`702fc42631de676637535042093b3fa2616f70e1701900b2bdcdecda9c27146c`.

Final evaluation report file SHA256:
`db80a05b127f9a90dde02676fe7b9c0e046a8b08163b213c389bff192cd4beae`.
Its canonical content hash is
`2cb4890cfcc3162fb3919a3d515f4aebbe08c2123706e250d5c3e38ec244944f`.
The report binds the calibration and evaluation bytes actually parsed, the
captured model/receipt bytes, and implementation hashes.

## Isolated query checks

```powershell
python -I -B .\frontier\offline_retrieval\navigator.py --model tfidf-v1 --query "UnifiedReceiptChain verify first_break SHA3-256 hash chain"
python -I -B .\frontier\offline_retrieval\navigator.py --model lsa-v1 --query "zyzzyva quetzalcoatl xylophonic flibbertigibbet"
```

The first query returned `RETRIEVED_LOCAL_EVIDENCE`, with top score
0.19713418185710907 against threshold 0.1172950491309166. Its first result was
the API excerpt in `corpus/kernels/README.md`. Receipt depth was 2,
with chain head
`98871e2d2b711c60c261fe72fce3ea028a0555fb1b060cf18842d827ff0f66e4`.

The second query returned `ABSTAIN_INSUFFICIENT_LOCAL_EVIDENCE`,
reason `NO_MODEL_VOCABULARY`, no results, and `kernel_called=false`.
Receipt depth was 1, with chain head
`903941434435e4d99c229cd565a25e866a874b0dd2f3a76a269eaa7cd050f3b8`.

On September 7, before this refresh, the README's original demo query,
"How does the receipt chain detect tampering?", ran successfully, but its first match repeated the query
in the README command example. That run establishes CLI and receipt operation
only; it is not additional retrieval-quality evidence.

After integration with the current repository base, the local CPU test suite
passed 162 cases with one CUDA case explicitly deselected; the source-binding
suite passed 10 tests. Ruff E9/F passed. Hosted CI, source merge, and provider
publication are separate release evidence, not inferred from those local runs.
