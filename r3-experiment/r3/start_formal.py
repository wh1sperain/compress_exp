"""One-time launch of the explicitly authorized, inference-only R3 scope."""
import json
import os
import subprocess
import time
from pathlib import Path
from .common import read, digest, code_hash, Blocked
from .runner import publish_once, load_plan
from .execution_acceptance import create
from .register_run import register

RUN_ID = 'r3-20261024-native-inference-v1'
AUTHORIZATION = '''用户要求删除冗余测试，将长程大幅削减为仅保证正常；随后明确：
准备启动正式实验，不启动答案解析和评分，你回顾主方案，充分调度GPU资源进行推理。实验完成后下载所有结果到本地。
本次按现有原生B1执行已冻结main+controls 21900次；不声称实现连续批处理或通过原完整认证。'''


def run(folder):
    folder = Path(folder); release = Path(__file__).resolve().parent.parent
    launch_path = folder/'telemetry/formal_inference_launch.json'
    import fcntl
    with (folder/'formal-start.lock').open('a') as start_lock:
        fcntl.flock(start_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if launch_path.exists(): raise Blocked('Existing formal launch; never start a duplicate')
        for name in ('production.lock', 'production-coordinator.lock', 'formal-lifecycle.lock'):
            with (folder/name).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        if compute.stdout.strip(): raise Blocked('GPU already has an active compute process')
        cfg = folder/'configs'/RUN_ID
        if cfg.exists(): raise Blocked('Existing immutable formal configuration; inspect before recovery')
        cfg.mkdir(parents=True)
        evidence = []
        # Copy completed summaries into this immutable execution snapshot.
        for name in ('official_followup_state', 'mini_recovery_state', 'reduced_stability_state',
                     'natural_integration_state', 'llava_integration_state'):
            source = folder/'telemetry'/(name+'.json'); value = read(source)
            if value.get('stage') != 'complete' or value.get('exit_code') != 0:
                raise Blocked('Required completed diagnostic missing: '+name)
            target = cfg/(name+'.json'); publish_once(target, value); evidence.append(target)
        supplemental = read(folder/'telemetry/qwen4_supplemental_audit_launch.json')
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
        profile = {'backend':'native-b1','dtype':'bfloat16','kv_dtype':'bfloat16','max_live_requests':1,
                   'persistent_model':True,'parallel_model_copies':False,'continuous_batching':False,
                   'performance_target_achieved':False,'gpu':gpu,'cpu_threads':4,
                   'prefill_policy':'complete native request','sampler':'request-local PCG64 and GPU inverse CDF',
                   'checkpoint_interval_tokens':128,'output_caps':{'llava7':2048,'mini8':2048,'qwen8':32768,'qwen4':32768},
                   'parsing_enabled':False,'scoring_enabled':False,'original_research_settings_unchanged':True,
                   'test_scope':'user-reduced checks; full native/engine certification not established'}
        profile_path = cfg/'engine_profile.json'; publish_once(profile_path, profile)
        spec = release/'round3_spec.json'
        manifest = folder/'manifests/frozen-inputs-v1/main_manifest.jsonl'
        acceptance = cfg/'execution_acceptance.json'
        create(folder,spec,manifest,profile_path,supplemental['out'],supplemental['status_path'],
               evidence,AUTHORIZATION,['Full 6000-record natural equivalence matrix not run',
               'Full 32K long-context numeric matrix replaced by two 4096-step stability checks',
               'Continuous batching/paged production backend not implemented'],acceptance)
        registration_dir = folder/'jobs'/RUN_ID
        receipt = register(folder,spec,manifest,profile_path,['main','controls'],None,None,registration_dir,
                           supplemental['out'],supplemental['status_path'],acceptance)
        for item in receipt['plans']: load_plan(item['path'],folder)
        output = folder/'raw'/RUN_ID; output.mkdir(parents=True)
        archive_dir = folder/'exports'/RUN_ID
        env = dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        with (output/'lifecycle-launch.log').open('xb') as log:
            child = subprocess.Popen(['/root/miniconda3/bin/python','-u','-m','r3.formal_lifecycle',
                '--folder',str(folder),'--registration',str(registration_dir/'registration.json'),
                '--output',str(output),'--archive-dir',str(archive_dir)],cwd=release,env=env,
                stdout=log,stderr=log,start_new_session=True)
        launch = dict(run_id=RUN_ID,pid=child.pid,release=str(release),code_sha256=code_hash(),
                      started_at=time.time(),production_started=True,planned_records=21900,
                      registration=str(registration_dir/'registration.json'),output=str(output),archive_dir=str(archive_dir),
                      inference_only=True,parsing_enabled=False,scoring_enabled=False,
                      full_native_engine_certification=False,backend='native-b1')
        publish_once(launch_path,launch)
        return launch


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args()
    print(json.dumps(run(a.folder)))
