#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ensemble_all_branches'/'inventory'
OUT.mkdir(parents=True,exist_ok=True)

def run(*args, binary=False):
    return subprocess.check_output(args,cwd=ROOT,stderr=subprocess.STDOUT,text=not binary)

def show(ref,path,binary=False):
    return run('git','show',f'{ref}:{path}',binary=binary)

subprocess.run(['git','fetch','--all','--prune'],cwd=ROOT,check=True)
refs=run('git','for-each-ref','--format=%(refname:short)','refs/remotes/origin').splitlines()
refs=sorted(r for r in refs if not r.endswith('/HEAD') and 'ensemble-all-branches-20260716' not in r)
current=run('git','branch','--show-current').strip()

TASK_PATTERNS=[
 re.compile(r'(?:^|/)task(\d{3})(?:/|\.|_|$)',re.I),
 re.compile(r'(?:^|/)task_(\d{3})(?:/|\.|_|$)',re.I),
]
def task_id(path):
    for rx in TASK_PATTERNS:
        m=rx.search(path)
        if m:return int(m.group(1))
    return None

def collect_numbers(obj,out):
    if isinstance(obj,dict):
        keys=set(obj)
        if keys & {'public_wrong','arc_gen_wrong','fresh_right','fresh_total','cost','points','memory','params','gain','exact'}:
            out.append(obj)
        for v in obj.values():collect_numbers(v,out)
    elif isinstance(obj,list):
        for v in obj:collect_numbers(v,out)

catalog={'generated_from':current,'refs':[],'candidates':[]}
seen=set()
for ref in refs:
    commit=run('git','rev-parse',ref).strip()
    files=run('git','ls-tree','-r','--name-only',ref).splitlines()
    json_paths=[p for p in files if p.lower().endswith('.json')]
    parsed_json={}
    for jp in json_paths:
        if not any(k in jp.lower() for k in ('valid','score','report','resume','manifest','checkpoint','promotion','exact')):
            continue
        try:
            parsed_json[jp]=json.loads(show(ref,jp))
        except Exception:
            pass
    onnx_paths=[p for p in files if p.lower().endswith('.onnx')]
    ref_info={'ref':ref,'commit':commit,'file_count':len(files),'onnx_count':len(onnx_paths),'task_ids':sorted({task_id(p) for p in onnx_paths if task_id(p) is not None})}
    catalog['refs'].append(ref_info)
    for p in onnx_paths:
        tid=task_id(p)
        if tid is None: continue
        raw=show(ref,p,binary=True)
        sha=hashlib.sha256(raw).hexdigest()
        key=(tid,sha)
        if key in seen:
            for c in catalog['candidates']:
                if c['task']==tid and c['sha256']==sha:
                    c['sources'].append({'ref':ref,'commit':commit,'path':p})
                    break
            continue
        seen.add(key)
        parent=str(Path(p).parent)
        evidence=[]
        evidence_paths=[]
        for jp,obj in parsed_json.items():
            jparent=str(Path(jp).parent)
            if jparent==parent or f'task{tid:03d}' in jp.lower() or f'"task": {tid}' in json.dumps(obj) or f'"task": "{tid:03d}"' in json.dumps(obj):
                found=[]; collect_numbers(obj,found)
                if found:
                    evidence.extend(found); evidence_paths.append(jp)
        catalog['candidates'].append({
            'task':tid,'sha256':sha,'bytes':len(raw),
            'sources':[{'ref':ref,'commit':commit,'path':p}],
            'evidence_paths':sorted(set(evidence_paths)),
            'evidence':evidence,
        })

for c in catalog['candidates']:
    vals=[]
    for e in c['evidence']:
        if not isinstance(e,dict):continue
        pw=e.get('public_wrong'); aw=e.get('arc_gen_wrong')
        fr=e.get('fresh_right'); ft=e.get('fresh_total')
        cost=e.get('cost'); points=e.get('points'); exact=e.get('exact')
        fresh_acc=(fr/ft) if isinstance(fr,(int,float)) and isinstance(ft,(int,float)) and ft else None
        vals.append({'public_wrong':pw,'arc_gen_wrong':aw,'fresh_right':fr,'fresh_total':ft,'fresh_accuracy':fresh_acc,'cost':cost,'points':points,'memory':e.get('memory'),'params':e.get('params'),'exact':exact,'gain':e.get('gain')})
    c['validation_summaries']=vals
    c['has_zero_wrong_evidence']=any(v['public_wrong']==0 and v['arc_gen_wrong']==0 for v in vals)
    c['best_reported_cost']=min((v['cost'] for v in vals if isinstance(v['cost'],(int,float))),default=None)
    c['best_fresh_accuracy']=max((v['fresh_accuracy'] for v in vals if v['fresh_accuracy'] is not None),default=None)

catalog['candidate_count']=len(catalog['candidates'])
catalog['task_count']=len({c['task'] for c in catalog['candidates']})
(OUT/'candidate_catalog.json').write_text(json.dumps(catalog,indent=2,sort_keys=True))
(OUT/'branches.txt').write_text('\n'.join(f"{r['ref']}\t{r['commit']}\tonnx={r['onnx_count']}\ttasks={len(r['task_ids'])}" for r in catalog['refs'])+'\n')
(OUT/'summary.json').write_text(json.dumps({'refs':len(catalog['refs']),'candidates':catalog['candidate_count'],'tasks':catalog['task_count'],'validated_candidates':sum(c['has_zero_wrong_evidence'] for c in catalog['candidates'])},indent=2))
print((OUT/'summary.json').read_text())
