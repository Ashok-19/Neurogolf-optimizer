#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, re, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ensemble_all_branches'/'candidate_bundle'
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)

def run(*args,binary=False):
    return subprocess.check_output(args,cwd=ROOT,stderr=subprocess.STDOUT,text=not binary)
def show(ref,path,binary=False):
    return run('git','show',f'{ref}:{path}',binary=binary)

subprocess.run(['git','fetch','--all','--prune'],cwd=ROOT,check=True)
refs=run('git','for-each-ref','--format=%(refname:short)','refs/remotes/origin').splitlines()
refs=sorted(r for r in refs if not r.endswith('/HEAD') and 'ensemble-all-branches-20260716' not in r)
rxs=[re.compile(r'(?:^|/)task(\d{3})(?:/|\.|_|$)',re.I),re.compile(r'(?:^|/)task_(\d{3})(?:/|\.|_|$)',re.I)]
def tid(path):
    for rx in rxs:
        m=rx.search(path)
        if m:return int(m.group(1))
    return None

records={}
for ref in refs:
    commit=run('git','rev-parse',ref).strip()
    files=run('git','ls-tree','-r','--name-only',ref).splitlines()
    for p in files:
        if not p.lower().endswith('.onnx'): continue
        task=tid(p)
        if task is None: continue
        raw=show(ref,p,binary=True); sha=hashlib.sha256(raw).hexdigest(); key=f'{task:03d}_{sha}'
        rec=records.setdefault(key,{'task':task,'sha256':sha,'bytes':len(raw),'sources':[],'artifact_files':[]})
        rec['sources'].append({'ref':ref,'commit':commit,'path':p})
        model_dir=OUT/f'task{task:03d}'/sha
        model_dir.mkdir(parents=True,exist_ok=True)
        model_path=model_dir/'model.onnx'
        if not model_path.exists(): model_path.write_bytes(raw)
        parent=str(Path(p).parent)
        prefixes=[parent+'/', str(Path(parent).parent)+'/' if Path(parent).name in {'work','experiments'} else '']
        for q in files:
            if q==p or q.lower().endswith('.onnx'): continue
            if not any(pref and q.startswith(pref) for pref in prefixes): continue
            if Path(q).suffix.lower() not in {'.json','.py','.md','.txt','.yaml','.yml'}: continue
            try: data=show(ref,q,binary=True)
            except Exception: continue
            if len(data)>2_000_000: continue
            rel=Path('evidence')/re.sub(r'[^A-Za-z0-9._/-]+','_',q)
            dest=model_dir/rel
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
            s=str(rel)
            if s not in rec['artifact_files']:rec['artifact_files'].append(s)
        needles=(f'task{task:03d}/',f'tasks/task{task:03d}/')
        for q in files:
            if not any(n in q.lower() for n in needles): continue
            if Path(q).name not in {'validate_task.py','arcgen_task.py','task.json','code_golf_solution.py'}: continue
            try:data=show(ref,q,binary=True)
            except Exception:continue
            rel=Path('taskkit')/re.sub(r'[^A-Za-z0-9._/-]+','_',q)
            dest=model_dir/rel
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            s=str(rel)
            if s not in rec['artifact_files']:rec['artifact_files'].append(s)

manifest={'refs':refs,'candidate_count':len(records),'task_count':len({r['task'] for r in records.values()}),'candidates':sorted(records.values(),key=lambda r:(r['task'],r['sha256']))}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
print(json.dumps({'refs':len(refs),'candidates':len(records),'tasks':manifest['task_count']},indent=2))
