import csv
import hashlib
import json
import os
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BATCH = PROJECT / 'teacher_pilot/natural_rollout_v1/shopping_batch_v1'
TOKENIZER = Path('/Users/lizizheng/Agent_Mem/models/Qwen3-Embedding-0.6B/tokenizer.json')
INPUTS = {}


def offline():
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    def deny(*args, **kwargs):
        raise RuntimeError('COST_AUDIT_NETWORK_FORBIDDEN')
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    raw = path.read_bytes()
    if not path.is_relative_to(HERE):
        INPUTS[str(path)] = sha(raw)
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return default


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    raw=path.read_bytes()
    if not path.is_relative_to(HERE):INPUTS[str(path)]=sha(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def save(name, data):
    path=HERE/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def jsonl(name, data):
    (HERE/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))


def csvout(name, data):
    if not data:return
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def compact(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def percentile(values,p):
    if not values:return None
    a=sorted(values);i=(len(a)-1)*p;lo=int(i);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(i-lo)


def stats(values):
    return {'n':len(values),'mean':sum(values)/len(values) if values else None,
            'p50':percentile(values,.5),'p90':percentile(values,.9),
            'p95':percentile(values,.95),'max':max(values) if values else None}
