# Measured local results

Measured on 2026-09-07 (America/New_York) using Python 3.11.9, NumPy 2.4.6,
scikit-learn 1.9.0, and PyTorch 2.11.0+cu128. Training, evaluation, and the query
checks below ran locally on CPU. The models were fitted over 113 chunks from
11 admitted sources; no duplicate source files were excluded. TF-IDF has
7,937 features and LSA has 112 dimensions.

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
| TF-IDF | 0.764960 | 0.812500 | 0.350000 | 0.770838 | 0.770833 | 0.779861 |
| LSA | 0.764960 | 0.812500 | 0.350000 | 0.770838 | 0.770833 | 0.779861 |
| BM25 | 0.772586 | 0.875000 | 0.375000 | 0.645838 | 0.770833 | 0.738194 |

R-precision values preserve the benchmark implementation's emitted rounding.
BM25 uses `szl-retrieval-bench` revision
`280a94c204ce67ffbb995d99bd2e92a917f913d2`. Exact benchmark module hashes are
recorded in `artifacts/evaluation_report.json`.

| Method | Answerable accepted | Answerable abstained | Unanswerable abstained | Unanswerable accepted | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 6/8 | 2/8 | 4/4 | 0/4 | 0.875000 |
| LSA | 5/8 | 3/8 | 4/4 | 0/4 | 0.812500 |
| BM25 | 8/8 | 0/8 | 4/4 | 0/4 | 1.000000 |

"Accepted" means the score passed the calibrated threshold; it does not mean
every returned chunk was relevant. On this small set BM25 has higher nDCG@5,
Recall@5, and answerable acceptance than either fitted model. TF-IDF and LSA
have higher MAP. These results do not establish general superiority for any
method. The LSA compression does not improve these ranking metrics and rejects
one additional answerable query compared with TF-IDF.

The evaluation status is `HOLD_EXTERNAL_EVALUATION_REQUIRED`.
The model and report explicitly record `publication_eligible=false`,
`evaluation_independently_adjudicated=false`, and
`general_retrieval_quality=false`. Receipts are `UNSIGNED_HONEST`.
No acceleration, clinical-validity, production-runtime, or general
out-of-distribution detection claim is made.

## Reproducibility

These commands ran successfully from the repository root:

```powershell
python -I -B .\frontier\offline_retrieval\train_retrieval.py
python -I -B .\frontier\offline_retrieval\train_retrieval.py
python -I -B .\frontier\offline_retrieval\evaluate_retrieval.py --bench-root ..\szl-retrieval-bench
```

The two consecutive training runs produced identical SHA256 values for all
seven training artifacts. This verifies reproducibility in the recorded local
environment; other dependency versions and hardware have not been compared.
Scoped `.gitattributes` rules retain LF source and metadata bytes in checkouts.
NumPy archives retain their existing Git LFS binary attributes.

| Artifact | SHA256, identical in both runs |
|---|---|
| `documents.json` | `17635e4016d1986b78008847ae3ca0391c64a97f8f9b05eabae9ddf27392fd95` |
| `lsa-components.npz` | `a54ce9e740b319458b147a6f18ea730054ddee731750d022da0ac70a1e938512` |
| `lsa-documents.npz` | `36272479874ac9d1fad99e6bc3a7740a0622d908799d7d7c466cbe139f162b06` |
| `model_manifest.json` | `3c6f99e2028d52526e038d524d45cea79082eb9fc131b80724fb3d90a227ce66` |
| `tfidf-documents.npz` | `f461a7d5afb5e2a9818b2e2dabd8a1f4683b77a6772a0a127e891fc93ad73c0e` |
| `training_receipt.json` | `53917a1d0eb316ed2f661f7c9afc8b77cf1bfa86e7ff44eb08b2a6c8e4fe4d10` |
| `vectorizer.json` | `be6dd6b2b4718b699a3e8fb64437af0c1f241ab75c1a9f07a91bb4f3336c89ce` |

The shared training receipt chain head is
`e16e0b1fcd36b27cf3b6f04b67bf0ed367d2056980400567afcbfb6c4017ad34`.

Final evaluation report file SHA256:
`c9981629e0639e08a72f5ed94618078dfb4d2039db0b8714250aa5779df945e2`.
Its canonical content hash is
`2d62b2f7a1cff524405483f2cd6d7b80018ab7aab025f392e012ca80e0e3e418`.
The report binds the calibration and evaluation bytes actually parsed, the
captured model/receipt bytes, and implementation hashes.

## Isolated query checks

```powershell
python -I -B .\frontier\offline_retrieval\navigator.py --model tfidf-v1 --query "UnifiedReceiptChain verify first_break SHA3-256 hash chain"
python -I -B .\frontier\offline_retrieval\navigator.py --model lsa-v1 --query "zyzzyva quetzalcoatl xylophonic flibbertigibbet"
```

The first query returned `RETRIEVED_LOCAL_EVIDENCE`, with top score
0.19683228433132172 against threshold 0.11848391592502594. Its first result was
the API excerpt in `corpus/kernels/README.md`; the chain-verification docstring
in `torch-ext/szl_kernels/_chain.py` was also returned. Receipt depth was 2,
with chain head
`39a1b27dd7a282cbc6993a579794ad3b3c2b91baaebfde6d882f820e1be19299`.

The second query returned `ABSTAIN_INSUFFICIENT_LOCAL_EVIDENCE`,
reason `NO_MODEL_VOCABULARY`, no results, and `kernel_called=false`.
Receipt depth was 1, with chain head
`6aac1486b00ffd81ffd5bb754e77b64efb1c4ffb6c74579a6e45d157898e6a84`.

The README's original demo query, "How does the receipt chain detect
tampering?", also ran successfully, but its first match repeated the query
in the README command example. That run establishes CLI and receipt operation
only; it is not additional retrieval-quality evidence.

Full integration tests and hosted CI are recorded separately after integration
with the current repository base.
