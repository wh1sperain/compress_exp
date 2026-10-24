"""Coordinator must distinguish execution success from scientific comparison."""
import json
import tempfile
import unittest
from pathlib import Path
from r3.parallel_verification import execution_result
from r3.common import digest


class ComparisonReceiptTests(unittest.TestCase):
    def test_exit_zero_with_failed_receipt_is_failure_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json'
            raw=b'{"status":"FAIL","metrics":{"tv":0.5}}\n'
            path.write_bytes(raw)
            result=execution_result('comparison',0,path)
            self.assertEqual(result['status'],'FAIL')
            self.assertEqual(result['exit_code'],1)
            self.assertEqual(result['process_exit_code'],0)
            self.assertEqual(result['comparison_status'],'FAIL')
            self.assertEqual(result['comparison_receipt_sha256'],digest(raw))
            self.assertEqual(path.read_bytes(),raw)

    def test_pass_requires_both_process_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json'
            path.write_text(json.dumps({'status':'PASS'}))
            self.assertEqual(execution_result('comparison',0,path)['status'],'PASS')
            failed=execution_result('comparison',9,path)
            self.assertEqual(failed['status'],'FAIL')
            self.assertEqual(failed['exit_code'],9)

    def test_missing_malformed_and_inconclusive_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json'
            self.assertEqual(execution_result('comparison',0,path)['exit_code'],1)
            self.assertFalse(path.exists())
            for raw in (b'not-json',b'[]',b'null',b'{}',b'{"status":"INCONCLUSIVE"}'):
                path.write_bytes(raw)
                result=execution_result('comparison',0,path)
                self.assertEqual(result['status'],'FAIL')
                self.assertEqual(result['exit_code'],1)
                self.assertEqual(path.read_bytes(),raw)

    def test_regular_command_preserves_exit_status(self):
        self.assertEqual(execution_result('unit-tests',0)['exit_code'],0)
        self.assertEqual(execution_result('unit-tests',2)['exit_code'],2)


if __name__=='__main__':unittest.main()
