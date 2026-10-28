"""Tables and eight plot families from immutable metrics, never invented outcomes."""
import argparse
import html
import json
import os
from pathlib import Path
from .common import read,rows,ROOT

def report(metrics,records,out):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=metrics['configurations'];figures=[]
    # Individual series retain model/task/seed/mode; no pooling of independent units.
    groups={}
    for r in cfg:
        key=tuple(r[k] for k in ('phase','model','dataset','decoding_mode','seed','method'))
        groups.setdefault(key,[]).append(r)
    def finish(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.svg'));plt.close(fig);figures.append(name)
    for field,name in [('ctir','01-retention-ctir'),('accuracy','02-retention-accuracy')]:
        fig,ax=plt.subplots(figsize=(11,6))
        for key,rr in groups.items():
            rr=sorted((r for r in rr if r.get(field) is not None),key=lambda r:r['keep_ratio_target'])
            if rr:
                ax.plot([r['keep_ratio_target'] for r in rr],[r[field] for r in rr],'.-',alpha=.6,label='/'.join(map(str,key)))
                if field=='ctir':
                    for r in rr:
                        ci=r.get('ctir_ci95')
                        if ci:ax.vlines(r['keep_ratio_target'],ci[0],ci[1],alpha=.3)
        ax.set(xlabel='Target visual retention',ylabel=field);finish(fig,name)
    fig,ax=plt.subplots(figsize=(12,max(4,len(cfg)*.15)))
    vals=np.array([[r['ctir'] if r['ctir'] is not None else np.nan] for r in cfg])
    if len(cfg):
        im=ax.imshow(vals,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=ax)
        ax.set_yticks(range(len(cfg)),[f"{r['model']}/{r['dataset']}/{r['method']}/{r['keep_ratio_target']}/{r['seed']}" for r in cfg],fontsize=6)
    ax.set_title('Observed reasoning CTIR (%)');finish(fig,'03-ctir-heatmap')
    def key(r):return tuple(r[k] for k in ('phase','model','sample_id','decoding_mode','seed'))
    base={key(r):r for r in records if r['method']=='Full'};pairs=[]
    for r in records:
        if r['method']!='Full' and key(r) in base and r.get('reasoning_tokens') is not None and base[key(r)].get('reasoning_tokens') is not None:pairs.append((base[key(r)],r))
    fig,ax=plt.subplots();ax.scatter([b['reasoning_tokens'] for b,c in pairs],[c['reasoning_tokens'] for b,c in pairs],s=5,alpha=.2)
    ax.set(xlabel='Full reasoning tokens',ylabel='Compressed reasoning tokens');finish(fig,'04-paired-lengths')
    fig,ax=plt.subplots();ratios=[c['reasoning_tokens']/b['reasoning_tokens'] for b,c in pairs if b['reasoning_tokens']>=32]
    if ratios:ax.hist(ratios,bins=50)
    ax.set(xlabel='Ratio (Full >=32 tokens)',ylabel='Paired observations; descriptive across conditions');finish(fig,'05-ratio-distribution')
    fig,ax=plt.subplots(figsize=(12,5))
    for i,r in enumerate(cfg):
        g=r['growth']['2'];v=g['rate']
        if v is not None:
            ax.bar(i,v);ax.vlines(i,*g['ci95'],color='black')
    ax.set(xlabel='Configuration index (see table)',ylabel='P(ratio >=2), Wilson 95% CI');finish(fig,'06-twofold-rate')
    fig,ax=plt.subplots()
    for i,g in enumerate(('CC','CW','WC','WW')):
        values=[r['correctness_groups'][g].get('ctir') for r in cfg];values=[v for v in values if v is not None]
        if values:ax.scatter([i]*len(values),values,alpha=.4)
    ax.set_xticks(range(4),('CC','CW','WC','WW'));ax.set_ylabel('Conditional CTIR per configuration');finish(fig,'07-correctness-groups')
    fig,ax=plt.subplots()
    for r in cfg:
        if r['ctir'] is not None and r['accuracy'] is not None and r['accuracy_full'] is not None:
            ax.scatter(100*(r['accuracy']-r['accuracy_full']),r['ctir'],s=15)
    ax.set(xlabel='Accuracy change (percentage points)',ylabel='CTIR (%)');finish(fig,'08-cost-quality')
    def table(items):
        return '<table><tr><th>Index</th><th>Record (null means unavailable)</th></tr>'+''.join('<tr><td>'+str(i)+'</td><td><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></td></tr>' for i,r in enumerate(items))+'</table>'
    body='<html><meta charset="utf-8"><title>R4扩展验证</title><style>body{max-width:1200px;margin:30px auto;font-family:sans-serif}img{max-width:100%}td{border:1px solid #ddd;padding:8px}pre{white-space:pre-wrap}</style><h1>R4扩展验证报告</h1><p>'+html.escape(metrics['note'])+'</p>'
    for title,items in [('共享Full基线',metrics.get('baselines',[])),('总体性能与长度分布',cfg),('六任务宏平均',metrics['macro']),('采样多种子',metrics['robustness'])]:body+='<h2>'+title+'</h2>'+table(items)
    for name in figures:body+='<h2>'+name+'</h2><img src="'+name+'.svg">'
    (out/'report.html').write_text(body+'</html>',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);p.add_argument('--records',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();report(read(a.metrics),rows(a.records),a.out)
