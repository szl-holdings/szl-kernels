# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Kernel estate catalog — import, call, or honest UNAVAILABLE.

This is the suite-side hub load. Every published SZL kernel is named here and
probed by actually calling ``selfcheck`` / ``rule_check`` when the package is
importable. Missing packages stay UNAVAILABLE (never fake-green). GPU cubins
are UNAVAILABLE unless CUDA is really present. joblib/pickle are not load paths.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# repo -> importable module + probe. Keep in lockstep with a11oy / szl-serve.
ESTATE: Tuple[Dict[str, str], ...] = (
    {
        "key": "szl-kernels",
        "module": "szl_kernels",
        "hub_id": "SZLHOLDINGS/szl-kernels",
        "probe": "selfcheck",
        "role": "unified governed-kernel suite + MiniEmbed",
    },
    {
        "key": "szl-governed-norm",
        "module": "szl_governed_norm",
        "hub_id": "SZLHOLDINGS/szl-governed-norm",
        "probe": "selfcheck",
        "role": "RMSNorm/LayerNorm + SHA3-256 receipts",
        "in_suite": "governed_rms_norm",
    },
    {
        "key": "szl-lambda-gate",
        "module": "szl_lambda_gate",
        "hub_id": "SZLHOLDINGS/szl-lambda-gate",
        "probe": "selfcheck",
        "role": "advisory Λ weighted-geometric-mean gate",
        "in_suite": "governed_lambda_gate",
    },
    {
        "key": "governed-inference-meter",
        "module": "governed_inference_meter",
        "hub_id": "SZLHOLDINGS/governed-inference-meter",
        "probe": "selfcheck",
        "role": "MEASURED joules; UNAVAILABLE without NVML",
        "in_suite": "governed_measure_energy",
    },
    {
        "key": "szl-receipt-attn",
        "module": "szl_receipt_attn",
        "hub_id": "SZLHOLDINGS/szl-receipt-attn",
        "probe": "selfcheck",
        "role": "tiled fused attention + receipts (Flash silhouette)",
    },
    {
        "key": "szl-maskmod",
        "module": "szl_maskmod",
        "hub_id": "SZLHOLDINGS/szl-maskmod",
        "probe": "selfcheck",
        "role": "score_mod + block-mask attention (Flex silhouette)",
    },
    {
        "key": "szl-block-kv",
        "module": "szl_block_kv",
        "hub_id": "SZLHOLDINGS/szl-block-kv",
        "probe": "selfcheck",
        "role": "paged KV cache + block-table receipts",
    },
    {
        "key": "YARQA-ATTN",
        "module": "yarqa_attn",
        "hub_id": "SZLHOLDINGS/YARQA-ATTN",
        "probe": "selfcheck",
        "role": "canal/compartment attention; not a Flash/Flex clone",
    },
    {
        "key": "szl-ouroboros",
        "module": "szl_ouroboros",
        "hub_id": "SZLHOLDINGS/szl-ouroboros",
        "probe": "selfcheck",
        "role": "bounded loop-tax accounting",
    },
    {
        "key": "szl-invariants",
        "module": "szl_invariants",
        "hub_id": "SZLHOLDINGS/szl-invariants",
        "probe": "selfcheck",
        "role": "8 falsifiable receipt invariants",
    },
    {
        "key": "szl-formulas",
        "module": "szl_formulas",
        "hub_id": "SZLHOLDINGS/szl-formulas",
        "probe": "selfcheck",
        "role": "21 canonical formulas; locked-proven exactly 8",
    },
    {
        "key": "szl-blocked",
        "module": "szl_blocked",
        "hub_id": "SZLHOLDINGS/szl-blocked",
        "probe": "selfcheck",
        "role": "honest BLOCKED first-class state",
    },
    {
        "key": "szl-govsign",
        "module": "szl_govsign",
        "hub_id": "SZLHOLDINGS/szl-govsign",
        "probe": "selfcheck",
        "role": "DSSE / in-toto governance signatures",
    },
    {
        "key": "szl-provctl",
        "module": "szl_provctl",
        "hub_id": "SZLHOLDINGS/szl-provctl",
        "probe": "selfcheck",
        "role": "provenance-DAG + in-toto/SLSA interop",
    },
    {
        "key": "szl-nemo",
        "module": "szl_nemo",
        "hub_id": "SZLHOLDINGS/szl-nemo",
        "probe": "rule_check",
        "role": "SOFTWARE/SURROGATE doctrine rule_check; joblib quarantined",
    },
    {
        "key": "szl-serve",
        "module": "szl_serve",
        "hub_id": "SZLHOLDINGS/szl-serve",
        "probe": "selfcheck",
        "role": "canonical serve recipe; GPU ROADMAP",
    },
)

_REQUIRED_KEYS = tuple(e["key"] for e in ESTATE)


def list_estate() -> List[Dict[str, str]]:
    return [dict(e) for e in ESTATE]


def cuda_status() -> Dict[str, Any]:
    """Honest CUDA probe. Never invents a device or a cubin."""
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            name = None
            try:
                name = str(torch.cuda.get_device_name(0))
            except Exception:
                name = "cuda:0"
            return {"status": "LIVE", "device": name}
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "GPU kernels stay ROADMAP; no fake CUDA",
        }
    return {
        "status": "UNAVAILABLE",
        "reason": "torch.cuda.is_available() is False",
        "note": "GPU kernels stay ROADMAP; no fake CUDA",
    }


def _extend_sys_path() -> None:
    extra = os.environ.get("SZL_KERNEL_PATHS", "")
    if not extra:
        return
    for raw in extra.split(os.pathsep):
        path = raw.strip()
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _summarize(result: Any) -> Any:
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        ok, violated = result
        return {"ok": ok, "violated": list(violated) if violated is not None else []}
    if isinstance(result, dict):
        out = {}
        for key in ("ok", "version", "label", "path", "lambda", "note", "checks"):
            if key in result:
                out[key] = result[key]
        if "ok" not in out and "arithmetic_ok" in result:
            out["ok"] = bool(result["arithmetic_ok"])
        return out or {"keys": sorted(result.keys())[:12]}
    return type(result).__name__


def _call_probe(mod: Any, probe: str) -> Any:
    if probe == "rule_check":
        fn = getattr(mod, "rule_check")
        return fn("hello", "this is MEASURED software, not a score")
    fn = getattr(mod, probe, None)
    if fn is None:
        raise AttributeError(f"{mod.__name__} has no {probe}()")
    return fn()


def _probe_in_suite(entry: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Call the suite's own wrappers so numeric members are LIVE without Hub fetch."""
    key = entry.get("in_suite")
    if not key:
        return None
    try:
        from . import (  # local suite numerics
            UnifiedReceiptChain,
            governed_lambda_gate,
            governed_measure_energy,
            governed_rms_norm,
        )
        import torch  # type: ignore
    except Exception:
        return None
    chain = UnifiedReceiptChain()
    if key == "governed_rms_norm":
        x = torch.randn(2, 8)
        governed_rms_norm(chain, x, eps=1e-6)
        ok, depth, brk = chain.verify()
        return {
            "status": "LIVE",
            "via": "szl_kernels.governed_rms_norm",
            "called": True,
            "chain_ok": bool(ok and depth == 1 and brk == -1),
        }
    if key == "governed_lambda_gate":
        gate = governed_lambda_gate(chain, torch.tensor([0.9, 0.8, 0.95]), threshold=0.5)
        return {
            "status": "LIVE",
            "via": "szl_kernels.governed_lambda_gate",
            "called": True,
            "advisory": bool(gate.get("advisory") is True),
        }
    if key == "governed_measure_energy":
        energy = governed_measure_energy(chain)
        return {
            "status": "LIVE",
            "via": "szl_kernels.governed_measure_energy",
            "called": True,
            "joules": energy.get("joules"),
            "label": energy.get("label"),
        }
    return None


def probe_member(entry: Dict[str, str]) -> Dict[str, Any]:
    rec = dict(entry)
    rec["joblib"] = "QUARANTINED"
    rec["pickle"] = "QUARANTINED"
    try:
        mod = importlib.import_module(entry["module"])
    except Exception as exc:
        in_suite = _probe_in_suite(entry)
        if in_suite is not None:
            rec.update(in_suite)
            rec["package"] = "UNAVAILABLE"
            rec["package_reason"] = f"{type(exc).__name__}: {exc}"
            return rec
        rec.update(
            {
                "status": "UNAVAILABLE",
                "called": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return rec
    try:
        result = _call_probe(mod, entry["probe"])
        rec.update(
            {
                "status": "LIVE",
                "via": f"{entry['module']}.{entry['probe']}",
                "called": True,
                "probe_result": _summarize(result),
            }
        )
        return rec
    except Exception as exc:
        rec.update(
            {
                "status": "UNAVAILABLE",
                "called": True,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return rec


def probe_estate() -> Dict[str, Any]:
    """Import+call every estate member. Missing stays UNAVAILABLE."""
    _extend_sys_path()
    kernels = [probe_member(dict(e)) for e in ESTATE]
    live = sum(1 for k in kernels if k.get("status") == "LIVE")
    return {
        "ok": live >= 1,
        "live": live,
        "enumerated": len(kernels),
        "cuda": cuda_status(),
        "joblib": "QUARANTINED",
        "pickle": "QUARANTINED",
        "lambda": "Conjecture 1 (advisory)",
        "kernels": kernels,
    }


def get_estate_member(key: str) -> Dict[str, str]:
    for entry in ESTATE:
        if entry["key"] == key:
            return dict(entry)
    raise KeyError(f"unknown estate member {key!r}; have {list(_REQUIRED_KEYS)}")
