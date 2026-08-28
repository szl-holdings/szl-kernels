# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""SZL-MiniEmbed — receipted distributional word-embedding TABLE.

This module loads the in-repo MiniEmbed artifact described by ``config.json``:

  * ``vocab.json``  — ``{"vocab": [term...], "index": {term: i}}``
  * ``vectors.npz`` — key ``vectors``, float32 ``[V, dim]`` (PPMI + TruncatedSVD)

It is a lookup table over terms already in that vocabulary. It is NOT a
transformer language model, NOT Chaski, and NOT Khipu. No throughput or
energy figures are produced or claimed. Λ remains Conjecture 1 (advisory).

Lookup / nearest-neighbour behaviour matches ``scripts/forge.py`` (cosine via
dot product on L2-normalized rows) and the file-hash contract in
``scripts/eval.py`` (SHA-256 of ``vectors.npz`` + ``vocab.json`` against
``TRAINING_RECEIPT.json``). Provenance receipts use the suite's existing
``UnifiedReceiptChain`` (SHA3-256) — this file does not invent a second chain.

Mirrored byte-for-byte at ``torch-ext/szl_kernels/miniembed.py`` (kernel-builder
source). Do not fork a second implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ._chain import UnifiedReceiptChain, tensor_digest

# Published SHA-256 of the SZL MiniEmbed v1 artifact (eval.py / TRAINING_RECEIPT).
PUBLISHED_SHA256: Dict[str, str] = {
    "vectors.npz": "053d839490ea95ed03dd1059f68474ef77cfcf04b2e727b29da9bc617abe4020",
    "vocab.json": "7862ebac9193d010606d89fb5aa6c046cbd638590d7a40c068c7bb60b3ed665e",
}

_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
_ARTIFACT_MARKERS = ("vocab.json", "vectors.npz", "config.json")
_KERNEL = "miniembed"
_LAMBDA_STATUS = "Conjecture 1 (open) — advisory only, NOT proven trust"


def _is_lfs_pointer(path: Path) -> bool:
    """True when ``path`` is a Git LFS pointer, not the real object bytes."""
    try:
        with path.open("rb") as handle:
            head = handle.read(len(_LFS_MAGIC) + 8)
    except OSError:
        return False
    return head.startswith(_LFS_MAGIC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_artifact_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Locate the repo-root MiniEmbed artifact directory.

    Walks up from this file (or ``start``) until ``vocab.json``, ``vectors.npz``,
    and ``config.json`` are all present — works from both
    ``build/torch-universal/szl_kernels/`` and ``torch-ext/szl_kernels/``.
    """
    if start is not None:
        root = Path(start).expanduser().resolve()
        missing = [name for name in _ARTIFACT_MARKERS if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"MiniEmbed artifact root {root} is missing {missing}"
            )
        return root
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if all((candidate / name).is_file() for name in _ARTIFACT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "SZL-MiniEmbed artifacts (vocab.json, vectors.npz, config.json) not found"
    )


def _hashes_from_receipt(payload: Mapping[str, Any]) -> Dict[str, str]:
    files = (payload.get("model") or {}).get("files") or {}
    out: Dict[str, str] = {}
    for name in PUBLISHED_SHA256:
        value = files.get(name)
        if isinstance(value, str) and len(value) == 64:
            out[name] = value.lower()
    return out


def _hashes_from_publication(payload: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    expected = payload.get("expected_artifact_sha256") or {}
    if isinstance(expected, Mapping):
        for name in PUBLISHED_SHA256:
            value = expected.get(name)
            if isinstance(value, str) and len(value) == 64:
                out[name] = value.lower()
    for item in (payload.get("source") or {}).get("files") or []:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        value = item.get("sha256")
        if path in PUBLISHED_SHA256 and isinstance(value, str) and len(value) == 64:
            out[path] = value.lower()
    for item in (payload.get("observed_hub_before_publication") or {}).get(
        "critical_artifacts"
    ) or []:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        value = item.get("sha256")
        if path in PUBLISHED_SHA256 and isinstance(value, str) and len(value) == 64:
            out[path] = value.lower()
    return out


def _load_json(path: Path) -> Optional[Any]:
    if not path.is_file() or _is_lfs_pointer(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


class MiniEmbed:
    """In-vocab lookup over the receipted SZL MiniEmbed table.

    HONESTY (doctrine v11): this is a distributional co-occurrence embedding
    table, not a generative model. Λ is untouched and remains Conjecture 1
    (advisory). Weights are the in-tree SZL MiniEmbed ``vectors.npz`` only —
    no third-party model weights are copied or loaded.
    """

    kind = "distributional_word_embedding_table"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        chain: Optional[UnifiedReceiptChain] = None,
    ) -> None:
        self.chain = chain if chain is not None else UnifiedReceiptChain()
        self.root: Optional[Path] = None
        self.config: Dict[str, Any] = {}
        self.vocab_terms: List[str] = []
        self.index: Dict[str, int] = {}
        self.vectors: Optional[np.ndarray] = None
        self.available = False
        self.unavailable_label: Optional[str] = None
        self.file_sha256: Dict[str, Optional[str]] = {
            "vectors.npz": None,
            "vocab.json": None,
        }
        self.expected_sha256: Dict[str, Dict[str, str]] = {
            name: {"published": digest} for name, digest in PUBLISHED_SHA256.items()
        }
        self._load(root)

    # -- load ----------------------------------------------------------------
    def _load(self, root: Optional[Union[str, Path]]) -> None:
        try:
            self.root = discover_artifact_root(root)
        except FileNotFoundError:
            self.unavailable_label = "UNAVAILABLE_MISSING_ARTIFACTS"
            return

        cfg = _load_json(self.root / "config.json")
        if isinstance(cfg, dict):
            self.config = dict(cfg)

        receipt = _load_json(self.root / "TRAINING_RECEIPT.json")
        if isinstance(receipt, Mapping):
            for name, digest in _hashes_from_receipt(receipt).items():
                self.expected_sha256[name]["TRAINING_RECEIPT.json"] = digest

        publication = _load_json(self.root / "publication.json")
        if isinstance(publication, Mapping):
            for name, digest in _hashes_from_publication(publication).items():
                self.expected_sha256[name]["publication.json"] = digest

        vocab_path = self.root / "vocab.json"
        vectors_path = self.root / "vectors.npz"

        if not vocab_path.is_file():
            self.unavailable_label = "UNAVAILABLE_MISSING_ARTIFACTS"
            return
        if _is_lfs_pointer(vocab_path):
            self.unavailable_label = "UNAVAILABLE_LFS"
            return
        self.file_sha256["vocab.json"] = _sha256_file(vocab_path)
        vocab_payload = _load_json(vocab_path)
        if not isinstance(vocab_payload, dict):
            self.unavailable_label = "UNAVAILABLE_BAD_VOCAB"
            return
        terms = vocab_payload.get("vocab")
        index = vocab_payload.get("index")
        if not isinstance(terms, list) or not isinstance(index, dict):
            self.unavailable_label = "UNAVAILABLE_BAD_VOCAB"
            return
        self.vocab_terms = [str(t) for t in terms]
        self.index = {str(k): int(v) for k, v in index.items()}

        if not vectors_path.is_file():
            self.unavailable_label = "UNAVAILABLE_MISSING_ARTIFACTS"
            return
        if _is_lfs_pointer(vectors_path):
            # Pointer hash is NOT the published object hash — refuse a fake pass.
            self.unavailable_label = "UNAVAILABLE_LFS"
            return
        self.file_sha256["vectors.npz"] = _sha256_file(vectors_path)
        try:
            with np.load(vectors_path, allow_pickle=False) as handle:
                if "vectors" not in handle.files:
                    self.unavailable_label = "UNAVAILABLE_BAD_VECTORS"
                    return
                table = np.array(handle["vectors"], dtype=np.float32, copy=True)
        except (OSError, ValueError, KeyError):
            self.unavailable_label = "UNAVAILABLE_BAD_VECTORS"
            return
        if table.ndim != 2:
            self.unavailable_label = "UNAVAILABLE_BAD_VECTORS"
            return
        want_v = int(self.config.get("vocab_size") or len(self.vocab_terms))
        want_dim = int(self.config.get("dim") or table.shape[1])
        if table.shape[0] != len(self.vocab_terms) or table.shape[0] != want_v:
            self.unavailable_label = "UNAVAILABLE_SHAPE_MISMATCH"
            return
        if table.shape[1] != want_dim:
            self.unavailable_label = "UNAVAILABLE_SHAPE_MISMATCH"
            return
        self.vectors = np.ascontiguousarray(table, dtype=np.float32)
        self.available = True
        self.unavailable_label = None

    # -- surface -------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab_terms)

    @property
    def dim(self) -> int:
        if self.vectors is not None:
            return int(self.vectors.shape[1])
        return int(self.config.get("dim") or 0)

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, term: object) -> bool:
        return isinstance(term, str) and term in self.index

    def __getitem__(self, term: str) -> np.ndarray:
        return self.lookup(term)

    def __repr__(self) -> str:
        status = "loaded" if self.available else (self.unavailable_label or "unavailable")
        return (
            f"MiniEmbed(vocab_size={self.vocab_size}, dim={self.dim}, "
            f"status={status!r})"
        )

    def _require_table(self) -> np.ndarray:
        if not self.available or self.vectors is None:
            raise RuntimeError(
                self.unavailable_label
                or "UNAVAILABLE_MISSING_ARTIFACTS"
            )
        return self.vectors

    def lookup(
        self,
        term: str,
        *,
        chain: Optional[UnifiedReceiptChain] = None,
    ) -> np.ndarray:
        """Return the float32 row for an in-vocab term. Raises KeyError if absent."""
        table = self._require_table()
        if term not in self.index:
            raise KeyError(f"term {term!r} is not in MiniEmbed vocab")
        idx = self.index[term]
        row = np.array(table[idx], dtype=np.float32, copy=True)
        self._emit_lookup(term, idx, row, chain)
        return row

    def encode(
        self,
        terms: Union[str, Sequence[str]],
        *,
        chain: Optional[UnifiedReceiptChain] = None,
    ) -> np.ndarray:
        """Encode in-vocab term(s). A string returns ``[dim]``; a sequence ``[N, dim]``."""
        if isinstance(terms, str):
            return self.lookup(terms, chain=chain)
        table = self._require_table()
        rows: List[np.ndarray] = []
        idxs: List[int] = []
        for term in terms:
            if term not in self.index:
                raise KeyError(f"term {term!r} is not in MiniEmbed vocab")
            idx = self.index[term]
            idxs.append(idx)
            rows.append(table[idx])
        if not rows:
            matrix = np.zeros((0, self.dim), dtype=np.float32)
        else:
            matrix = np.stack(rows, axis=0).astype(np.float32, copy=True)
        c = chain if chain is not None else self.chain
        c.emit(
            _KERNEL,
            "encode",
            {
                "n": int(matrix.shape[0]),
                "dim": int(self.dim),
                "indices": [int(i) for i in idxs],
                "out_digest": tensor_digest(torch.from_numpy(np.ascontiguousarray(matrix))),
                "kind": self.kind,
                "not_a_transformer_lm": True,
                "advisory": True,
                "lambda_status": _LAMBDA_STATUS,
            },
        )
        return matrix

    def neighbors(
        self,
        term: str,
        k: int = 6,
        *,
        chain: Optional[UnifiedReceiptChain] = None,
    ) -> Optional[List[Tuple[str, float]]]:
        """Cosine nearest neighbours, matching ``scripts/forge.py`` (None if OOV)."""
        table = self._require_table()
        if term not in self.index:
            return None
        idx = self.index[term]
        vec = table[idx]
        sims = table @ vec
        order = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for j in order:
            j = int(j)
            if j == idx:
                continue
            out.append((self.vocab_terms[j], round(float(sims[j]), 4)))
            if len(out) >= int(k):
                break
        c = chain if chain is not None else self.chain
        c.emit(
            _KERNEL,
            "neighbors",
            {
                "term": str(term),
                "index": int(idx),
                "k": int(k),
                "neighbour_count": int(len(out)),
                "kind": self.kind,
                "not_a_transformer_lm": True,
                "advisory": True,
                "lambda_status": _LAMBDA_STATUS,
            },
        )
        return out

    # -- receipts ------------------------------------------------------------
    def _emit_lookup(
        self,
        term: str,
        idx: int,
        row: np.ndarray,
        chain: Optional[UnifiedReceiptChain],
    ) -> None:
        c = chain if chain is not None else self.chain
        c.emit(
            _KERNEL,
            "lookup",
            {
                "term": str(term),
                "index": int(idx),
                "dim": int(row.shape[0]),
                "out_digest": tensor_digest(torch.from_numpy(np.ascontiguousarray(row))),
                "kind": self.kind,
                "not_a_transformer_lm": True,
                "not_chaski": True,
                "not_khipu": True,
                "advisory": True,
                "lambda_status": _LAMBDA_STATUS,
            },
        )

    def receipt_table(
        self,
        chain: Optional[UnifiedReceiptChain] = None,
    ) -> Dict[str, Any]:
        """Emit a SHA3-256 UnifiedReceiptChain record of the loaded table (or its honest unavailability)."""
        c = chain if chain is not None else self.chain
        attrs: Dict[str, Any] = {
            "kind": self.kind,
            "model": str(self.config.get("model") or "SZL-MiniEmbed"),
            "method": str(self.config.get("method") or ""),
            "not_a_transformer_lm": True,
            "not_chaski": True,
            "not_khipu": True,
            "advisory": True,
            "lambda_status": _LAMBDA_STATUS,
            "vocab_size": int(self.vocab_size),
            "dim": int(self.dim),
        }
        if self.available and self.vectors is not None:
            attrs["label"] = "LOADED"
            attrs["table_digest"] = tensor_digest(
                torch.from_numpy(np.ascontiguousarray(self.vectors))
            )
            attrs["vectors_sha256"] = self.file_sha256.get("vectors.npz")
            attrs["vocab_sha256"] = self.file_sha256.get("vocab.json")
        else:
            attrs["label"] = str(self.unavailable_label or "UNAVAILABLE")
            attrs["table_digest"] = None
            attrs["vectors_sha256"] = self.file_sha256.get("vectors.npz")
            attrs["vocab_sha256"] = self.file_sha256.get("vocab.json")
        return c.emit(_KERNEL, "table", attrs)

    def selfcheck(self) -> Dict[str, Any]:
        """Verify artifact SHA-256 against in-tree receipts. Never raises.

        A Git LFS pointer is reported as ``UNAVAILABLE_LFS`` — that is NOT a
        pass (the pointer bytes are not the published object).
        """
        checks: Dict[str, Any] = {
            "kind_is_table": self.kind == "distributional_word_embedding_table",
            "not_a_transformer_lm": True,
            "not_chaski": True,
            "not_khipu": True,
        }
        error = None
        label = self.unavailable_label or ("LOADED" if self.available else "UNAVAILABLE")
        try:
            honest_lfs = self.unavailable_label == "UNAVAILABLE_LFS"
            checks["honest_lfs"] = honest_lfs
            checks["vectors_available"] = bool(self.available)

            for name in ("vocab.json", "vectors.npz"):
                observed = self.file_sha256.get(name)
                wanted = self.expected_sha256.get(name) or {}
                unanimous = len(set(wanted.values())) == 1 if wanted else False
                checks[f"{name}_expected_agree"] = bool(unanimous)
                if name == "vectors.npz" and honest_lfs:
                    # Do not treat the pointer's SHA-256 as a match or a mismatch.
                    checks[f"{name}_sha256_match"] = False
                    checks[f"{name}_status"] = "UNAVAILABLE_LFS"
                    continue
                match = bool(
                    observed
                    and unanimous
                    and observed == next(iter(set(wanted.values())))
                )
                checks[f"{name}_sha256_match"] = match
                checks[f"{name}_status"] = (
                    "MATCHES" if match else ("MISSING" if observed is None else "MISMATCH")
                )

            rec = self.receipt_table()
            ok_chain, depth, brk = self.chain.verify()
            checks["table_receipt"] = bool(
                rec.get("kernel") == _KERNEL
                and rec.get("op") == "table"
                and ok_chain
                and brk == -1
                and depth >= 1
            )

            if self.available and "receipt" in self.index:
                row = self.lookup("receipt")
                checks["lookup_in_vocab"] = bool(
                    row.shape == (self.dim,) and row.dtype == np.float32
                )
                ok2, _, brk2 = self.chain.verify()
                checks["lookup_receipt"] = bool(ok2 and brk2 == -1)
            elif honest_lfs:
                checks["lookup_in_vocab"] = False
                checks["lookup_receipt"] = False
            else:
                checks["lookup_in_vocab"] = bool(self.available)
                checks["lookup_receipt"] = bool(self.available)

            if self.available:
                cfg_dim = int(self.config.get("dim") or 0)
                cfg_v = int(self.config.get("vocab_size") or 0)
                checks["config_shape"] = bool(
                    cfg_dim == self.dim and cfg_v == self.vocab_size
                )
        except Exception as exc:  # never raise from a health probe
            error = f"{type(exc).__name__}: {exc}"

        # A pass requires the real table + matching hashes. LFS is honest-unavailable.
        hash_ok = bool(
            checks.get("vocab.json_sha256_match")
            and checks.get("vectors.npz_sha256_match")
        )
        ok = bool(
            self.available
            and hash_ok
            and checks.get("table_receipt")
            and checks.get("lookup_receipt")
            and checks.get("lookup_in_vocab")
            and error is None
            and not checks.get("honest_lfs")
        )
        return {
            "ok": ok,
            "label": label,
            "available": bool(self.available),
            "vocab_size": int(self.vocab_size),
            "dim": int(self.dim),
            "checks": checks,
            "file_sha256": dict(self.file_sha256),
            "expected_sha256": {
                name: dict(sources) for name, sources in self.expected_sha256.items()
            },
            "chain_head": self.chain.head(),
            "error": error,
            "kind": self.kind,
        }


def expected_sha256_from_tree(root: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Union of published + in-tree TRAINING_RECEIPT / publication.json hashes."""
    embed = MiniEmbed(root=root)
    out: Dict[str, str] = {}
    for name, sources in embed.expected_sha256.items():
        if sources:
            out[name] = next(iter(sources.values()))
    return out


__all__ = [
    "MiniEmbed",
    "PUBLISHED_SHA256",
    "discover_artifact_root",
    "expected_sha256_from_tree",
]
