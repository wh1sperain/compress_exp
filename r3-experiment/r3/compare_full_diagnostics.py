"""Compare saved Full logits at shared real-image teacher-forcing probes."""
import argparse
from pathlib import Path
from .common import read,write,Blocked


def compare(root):
    import torch
    reference=root/'llava-Full';report={'native_gate_passed':False,'comparisons':{}}
    names=['fastv-FastVFull','llava-AgileFull','llava-VisionZipFull']
    if not (reference/'teacher_status.json').exists():raise Blocked('Reference not ready')
    ref=read(reference/'identity.json')
    a=torch.load(reference/'teacher_logits.pt',map_location='cpu',weights_only=True)
    for name in names:
        folder=root/name
        if not (folder/'teacher_status.json').exists():
            report['comparisons'][name]={'status':'INCONCLUSIVE','reason':'No completed teacher receipt'};continue
        if read(folder/'teacher_status.json')['status']!='PASS':
            report['comparisons'][name]={'status':'FAIL','reason':'Teacher trajectory failed'};continue
        identity=read(folder/'identity.json')
        keys=['model_revision','image_sha256','prompt_ids','teacher_prefix_sha256']
        if any(identity[k]!=ref[k] for k in keys):
            report['comparisons'][name]={'status':'FAIL','reason':'Different weights/image/prompt/prefix'};continue
        b=torch.load(folder/'teacher_logits.pt',map_location='cpu',weights_only=True)
        metrics=[]
        if set(a)!=set(b):raise Blocked('Missing logits probes')
        for step in sorted(a):
            x=a[step].double();y=b[step].double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise Blocked('Nonfinite logits')
            lp=x.log_softmax(-1);lq=y.log_softmax(-1);p=lp.exp();q=lq.exp()
            metrics.append({'step':step,'max_abs':float((x-y).abs().max()),
                'kl':float((p*(lp-lq)).sum()),'tv':float((p-q).abs().sum()/2),
                'argmax_equal':bool(x.argmax()==y.argmax())})
        report['comparisons'][name]={'status':'PASS' if all(m['kl']<.02 and m['tv']<.08 for m in metrics) else 'FAIL',
             'scope':'saved probe numerical tolerance only; not complete native gate','metrics':metrics}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    if Path(a.out).exists():raise Blocked('Refuse overwrite of comparison receipt')
    write(a.out,compare(Path(a.attempt)))
