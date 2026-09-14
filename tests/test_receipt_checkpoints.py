"""Execute the real receipt-chain source; no Hub, credentials or test-double loader."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'build/torch-universal/szl_kernels/_chain.py'
SPEC = importlib.util.spec_from_file_location('receipt_checkpoint_subject', SOURCE)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)
Chain = subject.UnifiedReceiptChain


def rehash(records):
    """Independent attacker-side recomputation: no subject hash helper used."""
    prev = '0' * 64
    for row in records:
        row['prev'] = prev
        body = {k: row[k] for k in ('seq', 'kernel', 'op', 'attrs', 'prev')}
        row['digest'] = hashlib.sha3_256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        prev = row['digest']
    return records


def example():
    c = Chain()
    for i in range(3):
        c.emit('test', 'operation', {'index': i, 'nested': [{'v': i}]})
    return c


def anchors(c):
    cp = c.checkpoint()
    return {'expected_head': cp['head'], 'expected_depth': cp['depth']}


class SnapshotTests(unittest.TestCase):
    def test_caller_attrs_detached(self):
        c = Chain(); attrs = {'nested': [{'value': 7}]}
        c.emit('k', 'op', attrs); original = c.to_json()
        attrs['nested'][0]['value'] = -1
        self.assertEqual(c.to_json(), original)
        self.assertEqual(c.verify(), (True, 1, -1))

    def test_returned_receipt_detached(self):
        c = Chain(); row = c.emit('k', 'op', {'nested': [7]}); original = c.to_json()
        row['attrs']['nested'][0] = -1; row['op'] = 'changed'
        self.assertEqual(c.to_json(), original)

    def test_tail_records_detached(self):
        c = example(); rows = c.tail(); original = c.to_json()
        rows[0]['attrs']['nested'][0]['v'] = 99
        rows[0]['digest'] = '1' * 64
        self.assertEqual(c.to_json(), original)
        self.assertEqual(c.verify(), (True, 3, -1))

    def test_tail_zero_is_empty(self):
        self.assertEqual(example().tail(0), [])

    def test_tail_bounds(self):
        for value in (-1, True, 1.0, '1', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                example().tail(value)

    def test_tail_positive_compatibility(self):
        self.assertEqual([r['seq'] for r in example().tail(2)], [1, 2])
        self.assertEqual(len(example().tail(100)), 3)

    def test_invalid_emit_does_not_advance(self):
        c = example(); cp = c.checkpoint()
        for attrs in ({'v': float('nan')}, {'v': float('inf')}, {'v': object()}, [], None):
            with self.subTest(attrs=repr(attrs)), self.assertRaises((TypeError, ValueError)):
                c.emit('k', 'op', attrs)
            self.assertEqual(c.checkpoint(), cp)

    def test_json_shape_and_hash_compatibility(self):
        c = Chain()
        attrs = {'nested': [1, None, {'unicode': 'Λ', 'zero': -0.0}], 'tuple': (1, 2)}
        with mock.patch.object(subject.time, 'time', return_value=1700000000.0):
            row = c.emit('norm', 'test', attrs)
        body = {'seq': 0, 'kernel': 'norm', 'op': 'test', 'attrs': attrs, 'prev': '0' * 64}
        expected = hashlib.sha3_256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(row['digest'], expected)
        self.assertEqual(set(row), {'seq', 'kernel', 'op', 'attrs', 'prev', 'digest', 'ts'})
        self.assertEqual(row['ts'], 1700000000.0)

    def test_concurrent_append_sequence(self):
        c = Chain()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: c.emit('thread', 'append', {'i': i}), range(80)))
        self.assertEqual(c.verify(**anchors(c)), (True, 80, -1))
        self.assertEqual(sorted(r['attrs']['i'] for r in c.tail(80)), list(range(80)))


class CheckpointTests(unittest.TestCase):
    def test_complete_export_matches_checkpoint(self):
        c = example()
        self.assertEqual(Chain.verify_json(c.to_json(), **anchors(c)), (True, 3, -1))

    def test_empty_checkpoint(self):
        c = Chain()
        self.assertEqual(c.checkpoint(), {'schema': 'szl.receipt-checkpoint/v1', 'depth': 0, 'head': '0' * 64})
        self.assertEqual(Chain.verify_json('[]', **anchors(c)), (True, 0, -1))

    def test_every_truncated_prefix_rejected_when_anchored(self):
        c = example(); rows = json.loads(c.to_json())
        for length in range(3):
            with self.subTest(length=length):
                blob = json.dumps(rows[:length])
                self.assertTrue(Chain.verify_json(blob)[0])  # Expected unanchored limit.
                self.assertEqual(Chain.verify_json(blob, **anchors(c)), (False, length, length))

    def test_rehashed_history_rejected_when_anchored(self):
        c = example(); rows = json.loads(c.to_json()); rows[0]['attrs']['index'] = 99
        changed = json.dumps(rehash(rows))
        self.assertTrue(Chain.verify_json(changed)[0])  # Hashes alone do not authenticate.
        self.assertFalse(Chain.verify_json(changed, **anchors(c))[0])

    def test_wrong_length_even_correct_head_rejected(self):
        c = example(); cp = anchors(c); cp['expected_depth'] += 1
        self.assertFalse(Chain.verify_json(c.to_json(), **cp)[0])

    def test_append_after_checkpoint_requires_new_anchor(self):
        c = example(); cp = anchors(c); c.emit('next', 'op', {})
        self.assertFalse(c.verify(**cp)[0])
        self.assertTrue(c.verify(**anchors(c))[0])

    def test_checkpoint_is_detached(self):
        c = example(); cp = c.checkpoint(); cp['head'] = 'a' * 64
        self.assertNotEqual(c.head(), cp['head'])

    def test_bad_checkpoint_arguments(self):
        cases = [dict(expected_head='a' * 64), dict(expected_depth=1),
                 dict(expected_head='A' * 64, expected_depth=1),
                 dict(expected_head='a' * 64, expected_depth=True),
                 dict(expected_head='a' * 64, expected_depth=-1),
                 dict(expected_head='a' * 64, expected_depth=0),
                 dict(expected_head='0' * 64, expected_depth=1)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Chain.verify_json('[]', **kwargs)

    def test_timestamps_are_not_authenticated(self):
        c = example(); rows = json.loads(c.to_json()); rows[0]['ts'] += 100
        self.assertTrue(Chain.verify_json(json.dumps(rows), **anchors(c))[0])

    def test_inconsistent_private_state_not_checkpointed(self):
        c = example(); c._records[0]['op'] = 'changed'
        with self.assertRaises(ValueError):
            c.checkpoint()


class JsonTests(unittest.TestCase):
    def test_malformed_or_nonlist_json_rejected(self):
        for value in ('', '{', '{}', 'null', 'true', '1', '"text"', b'[]', None, '[[1]]'):
            with self.subTest(value=value):
                self.assertFalse(Chain.verify_json(value)[0])

    def test_duplicate_keys_rejected(self):
        c = example(); blob = c.to_json().replace('"seq":0', '"seq":0,"seq":0', 1)
        self.assertFalse(Chain.verify_json(blob)[0])

    def test_nested_duplicate_keys_rejected(self):
        c = example(); blob = c.to_json().replace('"index":0', '"index":0,"index":0', 1)
        self.assertFalse(Chain.verify_json(blob)[0])

    def test_invalid_sequence_with_recomputed_digest_rejected(self):
        for value in (True, 0.0, -1, '0', 9):
            with self.subTest(value=value):
                rows = json.loads(example().to_json()); rows[0]['seq'] = value
                self.assertFalse(Chain.verify_json(json.dumps(rehash(rows)))[0])

    def test_missing_or_extra_record_fields_rejected(self):
        rows = json.loads(example().to_json())
        for key in set(rows[0]) - {'ts'}:
            sample = copy.deepcopy(rows); del sample[0][key]
            self.assertFalse(Chain.verify_json(json.dumps(sample))[0])
        rows[0]['approved'] = True
        self.assertFalse(Chain.verify_json(json.dumps(rows))[0])

    def test_timestamp_free_application_exports_remain_verifiable(self):
        # OfflineNavigator intentionally omits the unhashed timestamp. Keep the
        # existing six-field projection valid, including anchored verification.
        c = example(); rows = json.loads(c.to_json())
        for row in rows:
            row.pop('ts')
        blob = json.dumps(rows)
        self.assertEqual(Chain.verify_json(blob), (True, 3, -1))
        self.assertEqual(Chain.verify_json(blob, **anchors(c)), (True, 3, -1))

    def test_timestamp_free_prefix_still_requires_external_checkpoint(self):
        c = example(); rows = json.loads(c.to_json())
        for row in rows:
            row.pop('ts')
        blob = json.dumps(rows[:1])
        self.assertEqual(Chain.verify_json(blob), (True, 1, -1))
        self.assertEqual(Chain.verify_json(blob, **anchors(c)), (False, 1, 1))

    def test_optional_timestamp_never_allows_other_extra_fields(self):
        rows = json.loads(example().to_json())
        rows[0].pop('ts')
        rows[0]['approved'] = True
        self.assertFalse(Chain.verify_json(json.dumps(rows))[0])

    def test_present_timestamp_must_be_finite_numeric_metadata(self):
        for value in (None, True, 'yesterday', {}, [], float('nan'), float('inf')):
            with self.subTest(value=value):
                rows = json.loads(example().to_json()); rows[0]['ts'] = value
                self.assertFalse(Chain.verify_json(json.dumps(rows))[0])

    def test_nonfinite_values_rejected(self):
        for key in ('attrs', 'ts'):
            for value in (float('nan'), float('inf'), float('-inf')):
                rows = json.loads(example().to_json())
                rows[0][key] = {'value': value} if key == 'attrs' else value
                self.assertFalse(Chain.verify_json(json.dumps(rows))[0])
        rows = json.loads(example().to_json()); rows[0]['ts'] = 'REPLACE'
        self.assertFalse(Chain.verify_json(json.dumps(rows).replace('"REPLACE"', '1e999'))[0])

    def test_digest_prev_and_type_validation(self):
        for key, value in [('digest', None), ('digest', 'A' * 64), ('prev', 'x'), ('kernel', 1), ('op', False), ('attrs', []), ('ts', True)]:
            rows = json.loads(example().to_json()); rows[0][key] = value
            self.assertFalse(Chain.verify_json(json.dumps(rows))[0])

    def test_unrehashed_mutation_detected(self):
        rows = json.loads(example().to_json()); rows[1]['attrs']['index'] = -1
        self.assertEqual(Chain.verify_json(json.dumps(rows)), (False, 3, 1))

    def test_reorder_detected(self):
        rows = json.loads(example().to_json()); rows.reverse()
        self.assertFalse(Chain.verify_json(json.dumps(rows))[0])

    def test_energy_unavailable_preserved(self):
        c = Chain(); r = c.emit_energy({'label': 'UNAVAILABLE_NO_NVML', 'joules': None})
        self.assertIsNone(r['attrs']['joules'])
        self.assertEqual(c.verify(**anchors(c)), (True, 1, -1))

    def test_lambda_is_advisory(self):
        c = Chain(); r = c.emit_lambda(.8, .5, True, 3)
        self.assertIs(r['attrs']['advisory'], True)
        self.assertNotIn('production_authorization', r['attrs'])

    def test_actual_tensor_digest_and_norm_receipt(self):
        import struct
        import torch
        x = torch.tensor([[1., -2., 0.]], dtype=torch.float32)
        out = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = hashlib.sha3_256(struct.pack('<3q', *torch.round(out.flatten()*1000000).to(torch.int64).tolist())).hexdigest()
        c = Chain(); r = c.emit_norm('rms_norm', x, out, 1e-6)
        self.assertEqual(r['attrs']['out_digest'], expected)
        self.assertEqual(c.verify(**anchors(c)), (True, 1, -1))




class AtomicExportTests(unittest.TestCase):
    def test_export_and_checkpoint_capture_same_history(self):
        c = example()
        blob, cp = c.export_with_checkpoint()
        c.emit('new', 'later', {'v': 4})
        self.assertEqual(Chain.verify_json(blob, expected_head=cp['head'], expected_depth=cp['depth']), (True, 3, -1))
        self.assertFalse(c.verify(expected_head=cp['head'], expected_depth=cp['depth'])[0])

    def test_concurrent_exports_are_each_consistent(self):
        c = Chain()
        with ThreadPoolExecutor(max_workers=4) as pool:
            writes = [pool.submit(c.emit, 'worker', 'op', {'i': i}) for i in range(100)]
            captures = [pool.submit(c.export_with_checkpoint) for _ in range(20)]
            for task in captures:
                blob, cp = task.result()
                self.assertTrue(Chain.verify_json(blob, expected_head=cp['head'], expected_depth=cp['depth'])[0])
            for task in writes:
                task.result()
        self.assertEqual(c.count(), 100)

    def test_invalid_private_state_not_exported_as_checkpointed(self):
        c = example()
        c._records[0]['op'] = 'corrupted'
        with self.assertRaises(ValueError):
            c.export_with_checkpoint()


if __name__ == '__main__':
    unittest.main()
