#!/usr/bin/env python3
from __future__ import annotations
import base64, hashlib, json, re, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ensemble_all_branches'
PARTS = BASE / 'input'
OUT = BASE / 'inventory'
OUT.mkdir(parents=True, exist_ok=True)

def sh(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)

# Reconstruct and verify uploaded submission.
meta = json.loads((PARTS / 'manifest.json').read_text())
b64 = ''.join(p.read_text().strip() for p in sorted(PARTS.glob('submission.part*.b64')))
data = base64.b64decode(b64, validate=True)
assert len(data) == meta['source_bytes'], (len(data), meta['source_bytes'])
assert hashlib.sha256(data).hexdigest() == meta['source_sha256']
submission = BASE / 'submission-current.zip'
submission.write_bytes(data)
with zipfile.ZipFile(submission) as zf:
    names = zf.namelist()
    onnx_names = [n for n in names if re.fullmatch(r'task\d{3}\.onnx', n)]
    assert len(onnx_names) == 400, len(onnx_names)

subprocess.run(['git','fetch','--all','--prune'], cwd=ROOT, check=True)
refs = sh('git','for-each-ref','--format=%(refname:short)','refs/remotes/origin').splitlines()
refs = sorted(r for r in refs if not r.endswith('/HEAD'))
# Include current branch explicitly.
current = sh('git','branch','--show-current').strip()
if current and current not in refs:
    refs.append(current)

summary = {
    'submission': {
        'sha256': meta['source_sha256'],
        'bytes': meta['source_bytes'],
        'onnx_count': 400,
    },
    'python': subprocess.check_output(['python','-c','import sys; print(sys.version)'], text=True).strip(),
    'packages': subprocess.check_output(['python','-c','import onnx,onnxruntime,numpy; print(onnx.__version__, onnxruntime.__version__, numpy.__version__)'], text=True).strip(),
    'refs': [],
}

task_re = re.compile(r'(?:^|/)(?:task|task_?)(\d{3})(?:/|\.|_)', re.I)
for ref in refs:
    try:
        commit = sh('git','rev-parse',ref).strip()
        files = sh('git','ls-tree','-r','--name-only',ref).splitlines()
    except subprocess.CalledProcessError as exc:
        summary['refs'].append({'ref': ref, 'error': exc.output})
        continue
    onnx = [p for p in files if p.lower().endswith('.onnx')]
    validators = [p for p in files if p.endswith('validate_task.py') or p.endswith('/validate.py')]
    reports = [p for p in files if p.lower().endswith('.json') and any(k in p.lower() for k in ('valid','score','report','resume','manifest','checkpoint','promotion'))]
    builders = [p for p in files if p.lower().endswith('.py') and any(k in p.lower() for k in ('build','builder','prototype'))]
    tasks = {}
    for p in onnx:
        m = task_re.search('/'+p)
        if m:
            tasks.setdefault(m.group(1), []).append(p)
    summary['refs'].append({
        'ref': ref,
        'commit': commit,
        'file_count': len(files),
        'onnx_count': len(onnx),
        'validator_count': len(validators),
        'report_count': len(reports),
        'builder_count': len(builders),
        'task_count': len(tasks),
        'tasks': tasks,
        'validators': validators,
        'reports': reports,
    })

(OUT/'inventory.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
(OUT/'branches.txt').write_text('\n'.join(f"{x.get('ref')}\t{x.get('commit','')}\tonnx={x.get('onnx_count',0)}\ttasks={x.get('task_count',0)}" for x in summary['refs'])+'\n')
print(json.dumps({
    'refs': len(summary['refs']),
    'total_onnx_entries': sum(x.get('onnx_count',0) for x in summary['refs']),
    'submission_sha256': meta['source_sha256'],
}, indent=2))
