"""Local artifacts only. Secrets are read exclusively by the bounded live client."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
AUDIT=PROJECT/'agentmemory_api_cost_audit_v1'
PAIRED=PROJECT/'teacher_pilot/natural_rollout_v1/shopping_batch_v1/shopping_p_vs_rp_check'
TRAVEL=PROJECT/'teacher_demo_package/runtime/travel/raw_rollouts/travel_syn_2026091307_009'
PROVIDER=PROJECT/'runs/v4pro_tasks61_113_full_packet_20260909_r1/config.json'
INPUT_HASHES={}


def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):
    path=Path(path);raw=path.read_bytes()
    if not path.is_relative_to(HERE):INPUT_HASHES[str(path)]=sha(raw)
    return json.loads(raw)
def rows(path):
    path=Path(path);raw=path.read_bytes()
    if not path.is_relative_to(HERE):INPUT_HASHES[str(path)]=sha(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
def save(name,value):
    path=HERE/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def jsonl(name,values):
    (HERE/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values))
def verify_hashes(hashes):
    changed=[p for p,h in hashes.items() if not Path(p).exists() or sha(Path(p).read_bytes())!=h]
    if changed:raise RuntimeError('LOCKED_FILE_CHANGED: '+str(changed[:3]))
def forbid_network():
    import socket
    def deny(*args,**kwargs):raise RuntimeError('OFFLINE_NETWORK_FORBIDDEN')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
