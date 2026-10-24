"""Same-budget evidence intervention masks; no model output is used in selection."""
import numpy as np
from .common import digest,Blocked

def masks(uid,selected,values,bindings,scores):
    selected=set(selected);values=set(values);evidence=values|set(bindings);n=len(scores);universe=set(range(n))
    if not selected<=universe or not evidence<=universe or len(evidence)>len(selected):raise Blocked('Invalid evidence/index budget')
    removable=sorted(selected-evidence,key=lambda i:(scores[i],i));outside=sorted(universe-selected-evidence)
    result={}
    for label,needed in [('Value',values-selected),('Binding',evidence-selected)]:
        count=len(needed)
        if len(removable)<count or len(outside)<count:raise Blocked('Insufficient non-evidence controls')
        removed=set(removable[:count]);arm=label+'Random'
        rng=np.random.Generator(np.random.PCG64(int(digest('r3-rescue|'+uid+'|'+arm)[:16],16)))
        random_added=set(rng.choice(outside,count,replace=False).tolist()) if count else set()
        for name,added in [(label+'Rescue',needed),(arm,random_added)]:
            kept=sorted((selected-removed)|added);assert len(kept)==len(selected)
            result[name]={'selected_indices':kept,'removed':sorted(removed),'added':sorted(added),'zero_intervention':count==0}
    return result
