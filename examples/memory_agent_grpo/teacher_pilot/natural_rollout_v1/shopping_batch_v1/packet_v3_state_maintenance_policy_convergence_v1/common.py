from pathlib import Path

from ..trusted_packet_scaffold_convergence_v1.common import (
    HERE as PREVIOUS,
    digest,
    read,
    save,
    sha,
    write_rows,
)

HERE = Path(__file__).resolve().parent
REQUEST = Path('/Users/lizizheng/.codex/attachments/3891647a-63c2-4558-975e-11f42d90e11e/pasted-text.txt')
CAPSULES = PREVIOUS / 'archived_gate/capsules'
