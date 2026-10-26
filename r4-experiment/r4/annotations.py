"""Validate independent human evidence. Null/uncertain is never silently scored as correct."""
from .core import sha,select

ERRORS=('number_reading','spatial_localization','relation_binding','omission','unsupported_completion')
BEHAVIORS=('correct_rechecking','wrong_rechecking','question_rewrite','fruitless_trials','semantic_loop')

def validate(row, output_token_count):
    if len(set(row.get('reviewers',[])))<2 or not row.get('adjudicated'):
        raise ValueError('Independent human annotation/adjudication absent')
    if row.get('visual_error') not in (True,False,None):raise ValueError('Invalid error flag')
    events=row.get('events',[])
    for e in events:
        if e['type'] not in ERRORS or not 0<=e['start_token']<e['end_token']<=output_token_count:
            raise ValueError('Invalid token-grounded visual fact event')
        if not e.get('image_fact_id') or not e.get('quote'):raise ValueError('Evidence not grounded')
    if row.get('visual_error') is True and not events:raise ValueError('Error lacks visual fact evidence')
    if row.get('visual_error') is False and events:raise ValueError('Conflicting error annotations')
    return min((e['start_token'] for e in events),default=None)

def mechanism_panel(rows,salt='r4-m120-v1'):
    """6/task/stratum, four predeclared strata; shortage is reported, never filled by favorites."""
    strata=('new_error','full_error','correct_longer','correct_stable')
    from .core import TASKS
    selected=[];shortages=[]
    for task in TASKS:
        for stratum in strata:
            pool=[r for r in rows if r['task']==task and r['stratum']==stratum and r.get('adjudicated')]
            pool.sort(key=lambda r:sha([salt,r['uid']]))
            selected+=pool[:6]
            if len(pool)<6:shortages.append(dict(task=task,stratum=stratum,available=len(pool)))
    if len({r['uid'] for r in selected})!=len(selected):raise ValueError('Question in multiple strata')
    return {'selected':selected,'shortages':shortages,'actual_n':len(selected),'not_population_prevalence':True}
