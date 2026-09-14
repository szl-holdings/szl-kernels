# Receipt snapshots and externally retained checkpoints

This source change hardens the existing `UnifiedReceiptChain`; it is not a new
publisher, registry, model, scheduler, signature authority or production approval.
It affects both executable copies of `_chain.py` and preserves the legacy receipt
schema, digest encoding, and tensor-fingerprint function.

## Operational problem

Previously `emit()` retained the caller's nested `attrs` by reference and returned
the stored record itself. `tail()` copied the list but exposed its records. Mutating
these objects could corrupt already recorded history. `tail(0)` also returned the
entire history instead of no records.

Legacy verification checked linked hashes without enforcing ordinal sequence
numbers or strict JSON. More fundamentally, a valid prefix remained consistent
when its last records were removed. An attacker can also recompute an entire
unkeyed hash chain. A self-contained chain alone cannot authenticate its origin,
prove execution, or establish which complete history the verifier should expect.

## Changed behavior

`emit()` takes a detached JSON-value snapshot and returns a detached record.
`tail()` returns detached records, validates a nonnegative integer count and treats
zero as empty. JSON-compatible string-key attributes preserve the old digest
encoding. Input values must not be concurrently modified while their snapshot is
being made. These Python process controls do not defend against hostile code with
access to the process or private `_records`.

`verify()` and `verify_json()` still return `(ok, observed_depth, first_break)`.
They now check ordinal `seq`, strict record fields, field types and digest syntax.
The existing six-field application projection without `ts` remains supported;
when `ts` is present it must be finite numeric metadata, not a boolean. JSON parsing rejects duplicate keys and nonfinite
constants. Malformed exports return a failed tuple; invalid checkpoint arguments
raise `ValueError`. Extra fields and missing required hashed/digest fields are
rejected instead of silently ignored. No timestamp is invented for a timestamp-free
application export. Applications accepting external JSON must still enforce their own input
size and resource limits before this in-memory parser.

`checkpoint()` atomically obtains a valid chain's head and depth. For concurrent
writers, `export_with_checkpoint()` captures the matching export and checkpoint
under one lock:

```python
exported, checkpoint = chain.export_with_checkpoint()
# Store the checkpoint through the existing authenticated receipt/publication
# path, bound to run ID, source revision, policy and tenant as appropriate.
# Do not obtain the verifier's expected checkpoint from the untrusted export.
ok, depth, first_break = UnifiedReceiptChain.verify_json(
    exported,
    expected_head=checkpoint["head"],
    expected_depth=checkpoint["depth"],
)
```

The example checks consistency with an in-memory checkpoint. It does NOT implement
external custody or signing. The integration owner must store/authenticate that
checkpoint independently. A checkpoint delivered only alongside attacker-controlled
history provides no protection against replacing both. Neither new method writes
or transmits anything.

For checkpoint mismatch, `first_break` is the extent boundary
`min(observed_depth, expected_depth)`, not a claimed corrupt record. An unanchored
valid prefix deliberately continues to pass: changing that behavior would invent
a completeness proof. Matching an externally trusted checkpoint verifies the
expected recorded history under the hash assumptions, not that the recorded work
really happened.

## Compatibility and limits

Timestamp `ts` remains outside the legacy hashed body. Changing a finite timestamp
is not detected cryptographically. Tensor digests retain the existing rounded
float32 representation; equal rounded digests do not imply byte-identical original
tensors. No retrieval-quality, GPU-performance, energy measurement or Lambda theorem
claim changes.

The historical `corpus/kernels/` copy is left unchanged as a source input snapshot,
not silently rewritten as new runtime code. The two live copies remain byte-equal:
`build/torch-universal/szl_kernels/_chain.py` and
`torch-ext/szl_kernels/_chain.py`. The existing source-binding manifest already
names both. The corpus copy must not be substituted as executable patched code.

## Validation

Run from the repository root with its existing test dependencies:

```bash
python -m unittest discover -s tests -p 'test_receipt*.py' -v
python -O -m unittest discover -s tests -p 'test_receipt*.py' -v
```

The new tests execute the real local chain and retrieval implementations on small
synthetic CPU tensors. They exercise snapshot isolation, concurrent appends and
exports, duplicate keys, schema/sequence defects, external checkpoint mismatches,
valid-prefix truncation, fully rehashed history, ties, noncontiguous inputs, invalid
inputs and independent tensor-byte hashes. They do not use a simulated kernel
client and do not execute the Hugging Face `kernels` client at all.

Before source admission, run the existing complete repository suites and hosted
Python/platform matrix. Before runtime rollout, use the existing protected Forge
publisher with the new exact source revision, verify all declared provider bytes,
run actual Kernel Hub CPU and applicable GPU witnesses, and bind application
checkpoints to the existing trusted evidence path. Local tests are not those gates.
Do not change a production pin, drop review/check requirements, rewrite a historical
receipt or publish directly from this local patch.
