"""Inference-only lifecycle: registered workers, GPU telemetry, raw-result export.

This module never imports parsing or scoring code and never retries a failed job.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from .common import read, write, Blocked
from .runner import publish_once


def run(folder, registration, output, archive_dir):
    folder, registration, output, archive_dir = map(Path, (folder, registration, output, archive_dir))
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (folder/'formal-lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'lifecycle.json').exists():
            raise Blocked('Existing lifecycle must be inspected; no automatic relaunch')
        state = dict(pid=os.getpid(), stage='starting', started_at=time.time(),
                     inference_only=True, parsing_enabled=False, scoring_enabled=False,
                     registration=str(registration), output=str(output), archive_dir=str(archive_dir))
        def update():
            state['updated_at'] = time.time()
            write(output/'lifecycle.json', state)
        update()
        try:
            command = ['/root/miniconda3/bin/python', '-u', '-m', 'r3.run_coordinator',
                       '--folder', str(folder), '--registration', str(registration), '--out', str(output)]
            with (output/'coordinator.log').open('xb') as log, (output/'gpu_telemetry.jsonl').open('x') as gpu_log:
                child = subprocess.Popen(command, stdout=log, stderr=log)
                state.update(stage='generating', child_pid=child.pid); update()
                while child.poll() is None:
                    sample = {'time': time.time()}
                    try:
                        gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                        sample.update(gpu_csv=gpu.stdout.strip(), exit_code=gpu.returncode)
                    except Exception as exc:
                        sample['telemetry_error'] = type(exc).__name__
                    gpu_log.write(json.dumps(sample)+'\n'); gpu_log.flush()
                    update()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: pass
                state['coordinator_exit_code'] = child.returncode
            if child.returncode:
                raise Blocked('Formal inference failed; original outputs retained, no automatic retry')
            completion = read(output/'completion.json')
            if completion.get('stage') != 'generation_complete' or completion.get('exit_code') != 0:
                raise Blocked('Missing successful generation completion')
            state.update(stage='exporting', generation_completed_at=time.time()); update()
            from .export_results import export
            receipt = export(folder, registration, output, archive_dir, snapshot=False)
            state.update(stage='complete', export_receipt=receipt, completed_at=time.time()); update()
            publish_once(output/'lifecycle_completion.json', state)
        except BaseException as exc:
            state.update(stage='failed', error_type=type(exc).__name__, error=str(exc)[:1200]); update()
            publish_once(output/('lifecycle_failure-'+str(time.time_ns())+'.json'), state)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('folder', 'registration', 'output', 'archive-dir'): p.add_argument('--'+name, required=True)
    a = p.parse_args(); run(a.folder, a.registration, a.output, a.archive_dir)
