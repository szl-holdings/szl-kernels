#!/usr/bin/env python3
"""Forge SZL-MiniEmbed — a REAL trained word-embedding suite for SZLHOLDINGS/szl-kernels.

No gensim. Embeddings come from a term–term co-occurrence matrix (sliding window,
PPMI-weighted) reduced with sklearn TruncatedSVD — a classic, fully reproducible
distributional-semantics pipeline. The corpus is the SZL text estate:

  * SZLHOLDINGS/doctrine-v10-v11   (doctrine .md)
  * SZLHOLDINGS/rag-corpus-v1      (corpus.jsonl, ~948KB)
  * SZLHOLDINGS/thesis-corpus-v18  (thesis .tex chapters)
  * kernel-family READMEs          (this repo's own README + build metadata)

Ships vectors.npz + vocab.json + config.json. Evaluation is INTRINSIC ONLY:
nearest-neighbour sanity on ~15 doctrine terms (neighbour lists receipted).
No downstream benchmark is claimed. Seeded, receipted, reproducible.

Self-contained: resolves corpora from the repo's own dir when shipped in-repo
(corpus/ subdir), else from /tmp/corpus + /tmp/kernel-probe (forge-dev run)."""
import json, os, re, time, hashlib, platform, glob
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

SEED = 20260721
np.random.seed(SEED)
T0 = time.time()

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corpus_root(name, subdirs):
    """Prefer an in-repo bundled copy; else the forge-dev download location."""
    in_repo = os.path.join(_here, "corpus", name)
    if os.path.isdir(in_repo):
        return in_repo
    for sd in subdirs:
        if os.path.isdir(sd):
            return sd
    return None


DOCTRINE_DIR = corpus_root("doctrine-v10-v11", ["/tmp/corpus/doctrine-v10-v11"])
RAG_DIR = corpus_root("rag-corpus-v1", ["/tmp/corpus/rag-corpus-v1"])
THESIS_DIR = corpus_root("thesis-corpus-v18", ["/tmp/corpus/thesis-corpus-v18"])
KERNEL_DIR = corpus_root("kernels", ["/tmp/kernel-probe/szl-kernels", _here])

assert DOCTRINE_DIR and RAG_DIR, "doctrine + rag corpora required — refuse"

# ---------------------------------------------------------------------------
# 1. Load raw text from every source. Record byte sha256 of each source file.
# ---------------------------------------------------------------------------
docs = []            # list of raw text strings
source_files = []    # (path, sha256, n_chars)


def add_text(path, text):
    if not text or not text.strip():
        return
    docs.append(text)
    source_files.append((os.path.relpath(path, "/tmp"),
                         hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest(),
                         len(text)))


# doctrine markdown
for p in sorted(glob.glob(os.path.join(DOCTRINE_DIR, "*.md")) +
                glob.glob(os.path.join(DOCTRINE_DIR, "**", "*.md"), recursive=True)):
    try:
        add_text(p, open(p, encoding="utf-8", errors="ignore").read())
    except Exception:
        pass
# rag corpus.jsonl -> text field
rag_jsonl = os.path.join(RAG_DIR, "corpus.jsonl")
if os.path.isfile(rag_jsonl):
    parts = []
    for line in open(rag_jsonl, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            parts.append(json.loads(line).get("text", ""))
        except Exception:
            pass
    add_text(rag_jsonl, "\n".join(parts))
# thesis .tex chapters
if THESIS_DIR:
    for p in sorted(glob.glob(os.path.join(THESIS_DIR, "**", "*.tex"), recursive=True)):
        try:
            add_text(p, open(p, encoding="utf-8", errors="ignore").read())
        except Exception:
            pass
# kernel-family READMEs (this repo + its build metadata)
if KERNEL_DIR:
    for p in (sorted(glob.glob(os.path.join(KERNEL_DIR, "README.md"))) +
              sorted(glob.glob(os.path.join(KERNEL_DIR, "**", "*.md"), recursive=True)) +
              sorted(glob.glob(os.path.join(KERNEL_DIR, "**", "*.py"), recursive=True))):
        try:
            add_text(p, open(p, encoding="utf-8", errors="ignore").read())
        except Exception:
            pass

assert len(docs) >= 5, f"insufficient corpus ({len(docs)} docs) — refuse"

# ---------------------------------------------------------------------------
# 2. Tokenize. Keep alphabetic tokens + a few doctrine glyphs; lowercase; strip
#    latex/markdown control noise. Deterministic.
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]+|λ|ouroboros")
STOP = set("""the a an and or of to in is are was were be been being for on at by with as it its
this that these those from into than then so such not no nor but if while do does did done has have
had can could should would may might will shall must we you they he she i our your their his her them
us me my mine ours yours theirs which who whom whose what when where why how all any both each few more
most other some only own same too very s t just about above after again against because before below
between down during further here off out over under up once""".split())


def tokenize(text):
    text = re.sub(r"\\[a-zA-Z]+\{?|[{}$%#&_^~\\]", " ", text)  # strip latex/md control
    return [w for w in TOKEN_RE.findall(text.lower())
            if w not in STOP and len(w) > 2]


tokenized_docs = [tokenize(d) for d in docs]
all_tokens = [t for doc in tokenized_docs for t in doc]
freq = Counter(all_tokens)

MIN_COUNT = 5
vocab_terms = sorted([w for w, c in freq.items() if c >= MIN_COUNT])
MAX_VOCAB = 6000
if len(vocab_terms) > MAX_VOCAB:
    vocab_terms = [w for w, _ in Counter({w: freq[w] for w in vocab_terms}).most_common(MAX_VOCAB)]
    vocab_terms = sorted(vocab_terms)
vocab = {w: i for i, w in enumerate(vocab_terms)}
V = len(vocab)
assert V >= 200, f"vocab too small ({V}) — refuse"

# ---------------------------------------------------------------------------
# 3. Term–term co-occurrence (symmetric sliding window). PPMI weighting.
# ---------------------------------------------------------------------------
WINDOW = 5
cooc = defaultdict(float)
tok_total = 0
for doc in tokenized_docs:
    idx = [vocab[t] for t in doc if t in vocab]
    tok_total += len(idx)
    for i, wi in enumerate(idx):
        lo = max(0, i - WINDOW)
        hi = min(len(idx), i + WINDOW + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            wj = idx[j]
            cooc[(wi, wj)] += 1.0 / abs(j - i)  # distance-weighted

rows = np.fromiter((k[0] for k in cooc), dtype=np.int32, count=len(cooc))
cols = np.fromiter((k[1] for k in cooc), dtype=np.int32, count=len(cooc))
vals = np.fromiter((cooc[k] for k in cooc), dtype=np.float64, count=len(cooc))
C = csr_matrix((vals, (rows, cols)), shape=(V, V))

# PPMI: log( P(i,j) / (P(i)P(j)) ), clipped at 0
total = C.sum()
row_sum = np.asarray(C.sum(axis=1)).ravel()
col_sum = np.asarray(C.sum(axis=0)).ravel()
C = C.tocoo()
pmi_vals = np.log((C.data * total) / (row_sum[C.row] * col_sum[C.col]) + 1e-12)
pmi_vals = np.maximum(pmi_vals, 0.0)
PPMI = csr_matrix((pmi_vals, (C.row, C.col)), shape=(V, V))

# ---------------------------------------------------------------------------
# 4. TruncatedSVD -> dense L2-normalized embeddings.
# ---------------------------------------------------------------------------
DIM = 128
svd = TruncatedSVD(n_components=DIM, random_state=SEED, n_iter=10)
E = svd.fit_transform(PPMI)              # (V, DIM)
norms = np.linalg.norm(E, axis=1, keepdims=True)
norms[norms == 0] = 1.0
E = (E / norms).astype(np.float32)

# ---------------------------------------------------------------------------
# 5. INTRINSIC nearest-neighbour sanity on doctrine terms (present in vocab).
# ---------------------------------------------------------------------------
PROBE_TERMS = ["ouroboros", "governance", "governed", "receipt", "provenance",
               "invariant", "kernel", "energy", "lambda", "conjecture",
               "doctrine", "honest", "tamper", "verify", "chain",
               "loop", "signature", "attestation", "measured", "trust"]


def neighbors(term, k=6):
    if term not in vocab:
        return None
    v = E[vocab[term]]
    sims = E @ v
    order = np.argsort(-sims)
    out = []
    for idx in order:
        if idx == vocab[term]:
            continue
        out.append((vocab_terms[idx], round(float(sims[idx]), 4)))
        if len(out) >= k:
            break
    return out


neighbor_lists = {}
present = 0
for t in PROBE_TERMS:
    nb = neighbors(t)
    if nb is not None:
        neighbor_lists[t] = nb
        present += 1
assert present >= 12, f"only {present} probe terms in vocab — corpus too thin"

# ---------------------------------------------------------------------------
# 6. Save artifacts: vectors.npz + vocab.json + config.json.
# ---------------------------------------------------------------------------
out = _here
np.savez_compressed(os.path.join(out, "vectors.npz"), vectors=E)
with open(os.path.join(out, "vocab.json"), "w") as f:
    json.dump({"vocab": vocab_terms, "index": vocab}, f)
config = {
    "model": "SZL-MiniEmbed",
    "method": "term-term co-occurrence (distance-weighted, window=%d) -> PPMI -> TruncatedSVD" % WINDOW,
    "dim": DIM, "vocab_size": V, "min_count": MIN_COUNT, "window": WINDOW,
    "seed": SEED, "normalization": "L2 row-normalized",
    "files": {"vectors": "vectors.npz (key 'vectors', float32 [V,dim])",
              "vocab": "vocab.json ({'vocab':[term...], 'index':{term:i}})"},
}
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

vec_sha = hashlib.sha256(open(os.path.join(out, "vectors.npz"), "rb").read()).hexdigest()
vocab_sha = hashlib.sha256(open(os.path.join(out, "vocab.json"), "rb").read()).hexdigest()

# explained variance is a MEASURED intrinsic property of the SVD fit
explained = float(svd.explained_variance_ratio_.sum())

receipt = {
    "artifact": "SZLHOLDINGS/szl-kernels — SZL-MiniEmbed v1",
    "role": "intrinsic word-embedding suite over the SZL text estate — the kernel suite remains the primary artifact; embeddings are a companion, NOT a benchmark claim",
    "generator": {"script": "scripts/forge.py", "seed": SEED,
                  "method": config["method"], "no_gensim": True},
    "data": {
        "sources": ["SZLHOLDINGS/doctrine-v10-v11", "SZLHOLDINGS/rag-corpus-v1 (corpus.jsonl)",
                    "SZLHOLDINGS/thesis-corpus-v18", "szl-kernels family READMEs + build/*.py"],
        "n_documents": len(docs),
        "n_source_files": len(source_files),
        "total_tokens_in_window": int(tok_total),
        "vocab_size": V, "min_count": MIN_COUNT,
        "source_file_sha256": [{"path": p, "sha256": s, "n_chars": n} for p, s, n in source_files],
    },
    "model": {"type": "sklearn.TruncatedSVD over PPMI co-occurrence", "dim": DIM,
              "window": WINDOW, "params": {"n_components": DIM, "n_iter": 10, "random_state": SEED},
              "files": {"vectors.npz": vec_sha, "vocab.json": vocab_sha},
              "config": "config.json"},
    "metrics_MEASURED": {
        "vocab_size": V,
        "embedding_dim": DIM,
        "svd_explained_variance_ratio": round(explained, 4),
        "probe_terms_in_vocab": present,
        "intrinsic_nearest_neighbors": neighbor_lists,
        "claim_scope": "INTRINSIC SANITY ONLY — nearest-neighbour lists are the measured evidence; NO downstream/analogy benchmark score is claimed",
    },
    "environment": {"python": platform.python_version(),
                    "sklearn": __import__("sklearn").__version__,
                    "scipy": __import__("scipy").__version__,
                    "numpy": np.__version__, "host": "replit 2-vCPU container",
                    "wall_seconds": round(time.time() - T0, 1)},
    "honesty": "Every number above is MEASURED by this run. Embeddings are an intrinsic distributional artifact over the SZL corpus; no benchmark skill is claimed. The kernel suite stays the primary artifact. Λ untouched = Conjecture 1 (open).",
    "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(os.path.join(out, "TRAINING_RECEIPT.json"), "w") as f:
    json.dump(receipt, f, indent=2)

print(json.dumps({k: v for k, v in receipt["metrics_MEASURED"].items()
                  if k != "intrinsic_nearest_neighbors"}, indent=2))
for t in ["ouroboros", "governance", "receipt", "lambda"]:
    if t in neighbor_lists:
        print(f"  {t:12s} -> {[w for w,_ in neighbor_lists[t]]}")
print(f"docs={len(docs)} vocab={V} tokens={tok_total} wall={receipt['environment']['wall_seconds']}s")
