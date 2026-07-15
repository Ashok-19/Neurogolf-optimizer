#!/usr/bin/env python3
from __future__ import annotations
import json, re, subprocess, zipfile, io, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ensemble_all_branches'/'history_archives'
OUT.mkdir(parents=True,exist_ok=True)
def run(*a,binary=False):return subprocess.check_output(a,cwd=ROOT,text=not binary,stderr=subprocess.STDOUT)
subprocess.run(['git','fetch','--all','--tags','--prune'],cwd=ROOT,check=True)
lines=run('git','rev-list','--objects','--all').splitlines()
rows=[]
for line in lines:
    parts=line.split(' ',1)
    if len(parts)!=2:continue
    oid,path=parts
    low=path.lower()
    if not (low.endswith('.zip') or 'submission' in low or low.endswith('.tar.xz')):continue
    try:raw=run('git','cat-file','-p',oid,binary=True)
    except Exception:continue
    row={'oid':oid,'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if low.endswith('.zip'):
      try:
       with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist(); row['zip_entries']=len(names);row['task_onnx_count']=sum(bool(re.fullmatch(r'task\d{3}\.onnx',Path(n).name)) for n in names)
      except Exception as e:row['zip_error']=str(e)
    rows.append(row)
(OUT/'archives.json').write_text(json.dumps(rows,indent=2,sort_keys=True))
print(json.dumps({'matches':len(rows),'rows':rows},indent=2))
