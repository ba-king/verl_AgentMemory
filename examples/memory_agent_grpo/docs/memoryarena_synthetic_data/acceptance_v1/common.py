"""Local-only helpers for task generation and reference-aware acceptance."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
DATA=HERE.parent
PROJECT=DATA.parent.parent
WORKSPACE=next(p for p in HERE.parents if (p / 'pyproject.toml').is_file() and (p / 'examples/memory_agent_grpo').is_dir()).parent
ARENA=WORKSPACE/'MemoryArena'
AUDIT=DATA/'shopping_dependency_audit'
TRAVEL=DATA/'family_audits/travel'
SHOP=DATA/'shopping_v1'
REPLAY=HERE/'replays'
sys.path.insert(0,str(AUDIT))

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def save_rows(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in data))
def digest(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def require_preflight():
    if not read(HERE/'preflight.json')['passed']:raise RuntimeError('source_hash_preflight_failed')
def public_input(public,index):
    """Only current session. No private reference, previous transcript or future."""
    s=public['sessions'][index]
    out={'session_id':s['session_id'],'instruction':s['instruction']}
    if index==0 and public['task_family']=='travel':out['initial_observation']=public['initial_observation']
    return out

class PrefixGate:
    """Collection-only v0 rule: never replace an observed result by hidden gold."""
    def __init__(self):self.actual=[];self.stopped=False;self.failure=None
    def record(self,result,strict_pass):
        if self.stopped:raise RuntimeError('invalid_prefix_stopped')
        self.actual.append(result)
        if not strict_pass:self.stopped=True;self.failure='strict_step_failed'
    def next_allowed(self):return not self.stopped
