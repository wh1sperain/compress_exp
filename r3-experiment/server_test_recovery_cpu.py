import subprocess,json
from pathlib import Path
release=Path('/root/autodl-tmp/Compress/r3-releases/fb60dddd6256713c')
r=subprocess.run(['/root/autodl-tmp/Compress/runs/visual-compression-r3/environment/native-qwen/bin/python','-m','unittest','test_paged_recovery','test_batch_runner'],cwd=release,capture_output=True,text=True,timeout=60)
print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout[-2000:],stderr=r.stderr[-6000:])))
