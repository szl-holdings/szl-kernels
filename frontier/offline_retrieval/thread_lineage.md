# Thread lineage for the Offline Source Navigator

This record carries the user-visible intent of the September 2026 build thread
into the model package without copying private prompts, credentials, local key
material, or a raw conversation transcript.

The thread first produced Owned Agent Clinical Control
`2.4.0-synthetic-safe.3`, an executable Markdown/Python control payload for
offline synthetic fixtures. The source artifact was measured at 133,402 bytes
with SHA-256
`a5dae41ddd3d45c15ddbf015447081f07f50ad3c15a720669f091eeb9a5b95b6`.
It established durable state, exact schemas, bounded locks, encrypted Ed25519
keys, two-key authorization, immutable-content evidence checks, historical
receipts, and explicit boundaries around clinical validity and production
runtime claims.

The user then asked for a live audit of the SZL Holdings GitHub and Hugging
Face estates and for useful kernel and model work. This package applies that
request to a concrete public-interest problem: useful search over a small local
collection when network access or large-model compute is unavailable.

The following controls are carried into this package:

- refuse malformed, non-finite, empty, or unsupported input before recording a
  successful retrieval;
- bind model outputs to exact source and artifact hashes;
- emit a receipt that can be re-walked offline;
- return an explicit abstention when the calibrated local evidence is too weak;
- keep signatures, integrity, measured retrieval quality, clinical validity,
  and deployed runtime as separate claims;
- allow a solo operator to run the package without describing two keys as two
  independent human reviewers.

This lineage record is an implementation decision record. It is not a clinical
corpus, patient record, medical model, independent audit, or record of hidden
assistant instructions.
