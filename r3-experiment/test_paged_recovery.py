"""Focused regression checks for original-token recovery after backend migration.

CPU fake-backend tests only: these do not certify GPU numerical equivalence.
"""
import unittest
from unittest.mock import patch
import torch
import test_batch_scheduler as fixtures
from r3.common import Blocked, read, write
from r3.sampling import uniforms


class OriginalTokenRecoveryTest(unittest.TestCase):
    setUp = fixtures.SchedulerTest.setUp
    tearDown = fixtures.SchedulerTest.tearDown
    request = fixtures.SchedulerTest.request
    run_requests = fixtures.SchedulerTest.run_requests

    def enable_recovery(self):
        self.scheduler.profile.update(prefix_replay='original_tokens',
            recovery_id='test-recovery', recovery_code_sha256='test-source')

    def checkpoint(self, request, tokens):
        write(request[2], dict(job_key=request[1]['job_key'],
            seed=request[1]['seed'], generated_steps=len(tokens),
            generated_token_ids=tokens))

    def test_saved_tokens_are_forced_then_rng_continues_at_original_index(self):
        self.enable_recovery()
        request = self.request('restored', cap=5, seed=17)
        self.checkpoint(request, [1, 0])
        seen = []
        def sample(logits, u, **kwargs):
            seen.extend(u.tolist())
            return torch.tensor([1 if len(seen) == 1 else 2])
        result = self.run_requests([request], sample)[0][2]
        self.assertEqual(result['generated_token_ids'], [1, 0, 1, 2])
        self.assertEqual(seen, [float(x) for x in uniforms(17, 5)[2:4]])
        self.assertEqual(self.backend.batches, [[(5, 1)], [(6, 0)], [(7, 1)]])
        self.assertEqual(result['reconstructed_prefix_tokens'], 2)
        self.assertEqual(result['prefix_replay_policy'], 'original_tokens')
        self.assertEqual(result['execution_recovery_id'], 'test-recovery')
        self.assertEqual(read(request[2])['generated_token_ids'], [1, 0, 1, 2])

    def test_cap_length_prefix_never_invokes_sampler(self):
        self.enable_recovery()
        request = self.request('at-cap', cap=3)
        self.checkpoint(request, [1, 0, 1])
        with patch('r3.batch_scheduler.gpu_sample') as sample:
            result = list(self.scheduler.run([request], self.root / 'status.json'))[0][2]
        sample.assert_not_called()
        self.assertEqual(result['generated_token_ids'], [1, 0, 1])
        self.assertTrue(result['output_cap_reached'])
        self.assertFalse(result['eos_terminated'])

    def test_terminal_eos_prefix_never_invokes_sampler(self):
        self.enable_recovery()
        request = self.request('already-eos', cap=4)
        self.checkpoint(request, [1, 2])
        with patch('r3.batch_scheduler.gpu_sample') as sample:
            result = list(self.scheduler.run([request], self.root / 'status.json'))[0][2]
        sample.assert_not_called()
        self.assertEqual(result['generated_token_ids'], [1, 2])
        self.assertTrue(result['eos_terminated'])

    def test_mixed_new_and_restored_requests_keep_independent_rng_offsets(self):
        self.enable_recovery()
        restored = self.request('restored', cap=4, seed=17)
        fresh = self.request('fresh', cap=2, seed=19, position=11)
        self.checkpoint(restored, [1, 0])
        seen = []
        def sample(logits, u, **kwargs):
            seen.append(u.tolist())
            return torch.zeros(len(u), dtype=torch.long)
        outputs = self.run_requests([restored, fresh], sample)
        expected = [[float(x)] for x in uniforms(19, 2)]
        expected += [[float(x)] for x in uniforms(17, 4)[2:]]
        self.assertEqual(seen, expected)
        results = {job['job_key']: result for job, row, result in outputs}
        self.assertEqual(results['restored']['generated_token_ids'], [1, 0, 0, 0])
        self.assertEqual(results['fresh']['generated_token_ids'], [0, 0])
        self.assertEqual(len(self.backend.released), 2)

    def test_failure_during_replay_preserves_full_original_prefix(self):
        self.enable_recovery()
        request = self.request('interrupted', cap=5)
        self.checkpoint(request, [1, 0, 1])
        def fail(*args):
            raise RuntimeError('test replay interruption')
        self.backend.decode = fail
        with patch('r3.batch_scheduler.gpu_sample') as sample:
            with self.assertRaisesRegex(RuntimeError, 'test replay interruption'):
                list(self.scheduler.run([request], self.root / 'status.json'))
        sample.assert_not_called()
        self.assertEqual(read(request[2])['generated_token_ids'], [1, 0, 1])
        self.assertEqual(len(self.backend.released), 1)

    def test_interior_eos_rejected_before_prefill_or_sampling(self):
        self.enable_recovery()
        request = self.request('invalid-eos', cap=4)
        self.checkpoint(request, [0, 2, 1])
        with patch('r3.batch_scheduler.gpu_sample') as sample:
            with self.assertRaisesRegex(Blocked, 'interior EOS'):
                list(self.scheduler.run([request], self.root / 'status.json'))
        sample.assert_not_called()
        self.assertEqual(self.backend.states, [])
        self.assertEqual(read(request[2])['generated_token_ids'], [0, 2, 1])


if __name__ == '__main__':
    unittest.main()

