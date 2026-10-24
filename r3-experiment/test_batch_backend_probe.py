import unittest
from r3.batch_backend_probe import choose_rows, make_prefix, probe_steps
from r3.common import Blocked
from r3.manifest import TASKS


class BatchProbeTests(unittest.TestCase):
    def test_cross_task_distinct_images(self):
        rows=[dict(task=t,uid=t+str(i),image_sha256=t+str(i)) for t in TASKS for i in range(3)]
        picked=choose_rows(rows,4)
        self.assertEqual([r['task'] for r in picked],list(TASKS[:4]))
        self.assertEqual(len({r['image_sha256'] for r in picked}),4)
        with self.assertRaises(Blocked):choose_rows(rows,0)

    def test_forcing_excludes_eos(self):
        class Tokenizer:
            def encode(self,*args,**kwargs):return [1,2,3]
        prefix=make_prefix(Tokenizer(),[2],5)
        self.assertEqual(prefix,[1,3,1,3,1])
        with self.assertRaises(Blocked):make_prefix(Tokenizer(),[1,2,3],5)

    def test_long_probe_memory_bounded(self):
        self.assertEqual(probe_steps(32),set(range(32)))
        self.assertEqual(probe_steps(4096),{0,1,31,127,1023,4095})


if __name__=='__main__':unittest.main()
