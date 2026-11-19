"""Check that likelihood comparisons retain one version of their inputs.

The small files stand for a measured vector and its hash manifest. These
tests exercise replacement during a comparison without requiring CAMB or
the compiled likelihood. Numerical process checks live in the companion
command-line program.
"""

import json
from pathlib import Path
import tempfile
import unittest

from check_likelihood_state import copy_inputs, file_hash, verify_snapshot


class InputSnapshotTests(unittest.TestCase):
    """Keep copied data independent and reject files inconsistent with a manifest."""

    def setUp(self):
        """Create a private project with one frozen data file and its hash."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.source = self.project / "tests" / "frozen" / "data.txt"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("0 1.25\n")
        self.manifest = self.project / "tests" / "manifest_sha256.json"
        self.write_manifest()
        helper = self.project / "tests" / "cocoa_test_utils.py"
        helper.write_text('"""Private input-snapshot test file."""\n')
        self.destination = self.root / "copy"

    def tearDown(self):
        """Release only the temporary files owned by this test."""
        self.temporary.cleanup()

    def write_manifest(self):
        """Record the current source data as the project's frozen version."""
        record = {"files": {"frozen/data.txt": file_hash(self.source)}}
        self.manifest.write_text(json.dumps(record))

    def test_source_refreeze_does_not_change_snapshot(self):
        """A later reference regeneration cannot change the copied data."""
        expected = copy_inputs(project=self.project, destination=self.destination)
        self.source.write_text("0 9.75\n")
        self.write_manifest()
        actual = verify_snapshot(tests=self.destination)
        self.assertEqual(actual, expected)
        copied = self.destination / "frozen" / "data.txt"
        self.assertEqual(copied.read_text(), "0 1.25\n")

    def test_inconsistent_source_is_rejected(self):
        """A changed data file cannot enter a copy with an older manifest."""
        self.source.write_text("0 9.75\n")
        with self.assertRaisesRegex(RuntimeError, "expected"):
            copy_inputs(project=self.project, destination=self.destination)

    def test_changed_snapshot_is_rejected(self):
        """A modified private data file fails before a numerical comparison."""
        copy_inputs(project=self.project, destination=self.destination)
        copied = self.destination / "frozen" / "data.txt"
        copied.write_text("0 9.75\n")
        with self.assertRaisesRegex(RuntimeError, "expected"):
            verify_snapshot(tests=self.destination)


if __name__ == "__main__":
    unittest.main()
