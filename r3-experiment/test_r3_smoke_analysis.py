import json
import tempfile
import unittest
from pathlib import Path
from r3.common import Blocked
from r3.natural_smoke_analysis import collect_attempts


class SmokeAggregation(unittest.TestCase):
    def fixture(self,root):
        attempt=Path(root)/'attempt';(attempt/'key').mkdir(parents=True)
        job={'job_key':'key','model':'qwen8','condition':'Full','stage':'natural_certification',
             'side':'candidate','record_complete':False,'uid':'u'}
        record=dict(job,record_complete=True,generated_token_ids=[1])
        (attempt/'key/result.json').write_text(json.dumps(record))
        (attempt/'integration_smoke.json').write_text(json.dumps({'completed':1,'exit_code':0,'stage':'integration_smoke_complete'}))
        return attempt,{'key':job}
    def test_complete_smoke_not_full_matrix(self):
        with tempfile.TemporaryDirectory() as root:
            attempt,expected=self.fixture(root)
            files,receipts,counts=collect_attempts([attempt],'qwen8','Full',expected)
            self.assertEqual(len(files),1);self.assertEqual(counts,{'candidate':1})
            self.assertEqual(receipts[0]['records'],1)
            with self.assertRaises(Blocked):collect_attempts([attempt,attempt],'qwen8','Full',expected)
    def test_failed_or_partial_attempt_refused(self):
        with tempfile.TemporaryDirectory() as root:
            attempt,expected=self.fixture(root)
            (attempt/'key/failure-original.json').write_text('{}')
            with self.assertRaises(Blocked):collect_attempts([attempt],'qwen8','Full',expected)
        with tempfile.TemporaryDirectory() as root:
            attempt,expected=self.fixture(root)
            (attempt/'integration_smoke.json').write_text(json.dumps({'completed':2,'exit_code':0}))
            with self.assertRaises(Blocked):collect_attempts([attempt],'qwen8','Full',expected)


if __name__=='__main__':unittest.main()
