import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.wave4 import pulse  # noqa: E402


class Wave4Tests(unittest.TestCase):
    def test_source_complete_metal_blocked(self) -> None:
        report = pulse()
        self.assertEqual(report["source"], "COMPLETE")
        self.assertEqual(report["metal"], "BLOCKED_NO_METAL")
        self.assertFalse(report["ready"])
        self.assertIsNone(report["winner"])
        self.assertFalse(report["acceleration_story"])


if __name__ == "__main__":
    unittest.main()
