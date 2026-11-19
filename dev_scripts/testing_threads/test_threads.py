"""Check thread-count selection before importing the numerical libraries."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestValidationThreads(unittest.TestCase):
    """The default remains four; an explicit multi-threaded override is allowed."""

    def run_import(self, value):
        """Import the shared test machinery in a process with the chosen override."""
        environment = dict(os.environ)
        environment.pop("COCOA_TEST_OMP_THREADS", None)
        if value is not None:
            environment["COCOA_TEST_OMP_THREADS"] = value
        core = Path(__file__).resolve().parents[2]
        environment["PYTHONPATH"] = str(core)
        return subprocess.run(
            args=[sys.executable, "-B", "-c",
                  "import cocoa_testing as c; print(c.REQUIRED_OMP_THREADS)"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_and_override(self):
        """Use four by default and honor an explicit six-thread run."""
        for value, expected in [(None, "4"), ("6", "6")]:
            with self.subTest(value=value):
                result = self.run_import(value=value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_invalid_count(self):
        """A race test requires an integer count of at least two."""
        for value in ["0", "1", "-1", "2.5", "nan", ""]:
            with self.subTest(value=value):
                result = self.run_import(value=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("choose an integer >= 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
