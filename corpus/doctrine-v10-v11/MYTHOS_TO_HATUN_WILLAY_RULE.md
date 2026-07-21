# Mythos → Hatun-Willay rename rule
**Locked:** 2026-05-31 21:17 EDT
**Source:** Founder verbatim — "Putnam / Jarvis / Mythos / Bo11y Computacenter yes these can be skipped if mythos shows up just scratch but if it's fully options. And real a name it Kent g in quecha"

## Decision tree per banned token

| Token | If fully real + operational | If not fully real |
|---|---|---|
| Putnam | Keep as "Putnam 2025 LLM benchmark harness" (already canonical at `agi-forecast`) — it IS real (10/12 structure, 2/12 Lean-discharged) | Skip |
| Jarvis | Skip — legacy identity, no real code path | Skip |
| Mythos | **RENAME to Hatun-Willay** (Quechua: "the great telling / great proclamation") — founder pick 2026-05-31 21:20 EDT, keeps Hatun-lineage continuity with Hatun-Doctrine (platform PR #269) | Skip |
| Bo11y | Skip — legacy identity | Skip |
| Computacenter | NEVER MENTION — founder hard rule "don't bring up comp center it's our secret" | Skip |

## Rename mechanic for Mythos

Wherever an agent discovers a Mythos-named module/package/file that's confirmed working + operational:
1. Do NOT delete it
2. Rename the identifier to `HatunWillay` (CamelCase), `hatun_willay` (snake), `hatun-willay` (kebab), or `Hatun-Willay` (display)
3. Add a one-line provenance comment: "// Renamed from 'Mythos' on 2026-05-31 per founder doctrine v8 — same code path, Quechua lineage."
4. Preserve all git history (use `git mv` not delete+add)
5. Update any cross-references (imports, docs, CITATION.cff) atomically in the same PR

## Doctrine v8 status
Doctrine v7 banned Mythos outright. Doctrine v8 (proposed by PhD-Orchestrator synthesis pending) allows renamed instances. This file is the authoritative source until v8 is locked.

— Yachay (Perplexity Computer Agent), under CTO authority
