"""Repair only replay metadata; reuse the frozen parser, apply and auditor.

The previous collector remains sealed. No facts or Teacher deltas are repaired.
An applied packet is compared against a fresh deterministic replay, not itself.
"""
import copy

from .common import digest
from ..packet_interface_v3_minimal_validation import interface_v3 as V3
from ..trusted_packet_scaffold_convergence_v1 import kernel as frozen

evaluate = frozen.evaluate
model_input = frozen.model_input


def replay_metadata(previous, delta, applied, source_registry):
    replayed = V3.apply(copy.deepcopy(previous), copy.deepcopy(delta),
                        copy.deepcopy(source_registry), init=False)
    return {
        'replay_equal': applied == replayed,
        'applied_packet_sha256': digest(applied),
        'replayed_packet_sha256': digest(replayed),
        'previous_packet_sha256': digest(previous),
        'delta_sha256': digest(delta),
        'source_registry_sha256': digest(source_registry),
    }


def apply_response(row, response, trusted):
    row.update(schema_valid=False, apply_valid=False, replay_equal=False)
    frozen.apply_response(row, response, trusted)
    if row['apply_valid']:
        try:
            metadata = replay_metadata(row['previous_packet'], row['parsed_packet_delta'],
                                       row['applied_packet'], trusted['source_ownership_registry'])
            row['replay_metadata'] = metadata
            row['replay_equal'] = metadata['replay_equal']
        except Exception as exc:
            row['replay_error'] = type(exc).__name__ + ': ' + str(exc)
    return row
