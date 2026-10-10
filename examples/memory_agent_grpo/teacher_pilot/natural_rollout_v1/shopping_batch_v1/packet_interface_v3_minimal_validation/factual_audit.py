"""Finite external factual audit, deliberately distinct from V3 binding/scope.

Predicate/value entailment and per-subject completion remain external concerns.
No fact-to-subject or text-to-required-subject-set lookup occurs here.
"""
from .common import digest


def claims_for(audit, evidence):
    entry = audit['literal_interpretations'].get(digest(evidence['fact']))
    if entry is None or entry['text'] != evidence['fact']:
        return None
    return [{**c, 'subject_ref': evidence['subject_ref']} for c in entry['claims']]


def key(claim):
    return tuple(claim[k] for k in ('subject_ref', 'predicate', 'value'))


def supported(audit, trusted, claim, refs):
    if claim['kind'] == 'derived':
        for derivation in audit['derived_claims']:
            if key(derivation) != key(claim) or derivation['required_memory_ref'] not in refs:
                continue
            if all(any(key(p) == key(c) and c['source_ref'] in refs
                       for c in trusted['allowed_claims_up_to_event']) for p in derivation['premises']):
                return True
        return False
    return any(key(c) == key(claim) and c['source_ref'] in refs for c in trusted['allowed_claims_up_to_event'])


def completion_receipts(audit, trusted):
    """Factual completion receipts; no scope refs, need text or Packet is read.

Tests can supply other receipts to the structural scope gate. The bounded
predicate criteria describe e.g. literal+price, not the expected candidate set.
This does NOT imply arbitrary needs can be auto-scored in production.
"""
    receipts = {}
    for nid, criterion in audit['need_completion_criteria'].items():
        receipts[nid] = {}
        for subject in trusted['subject_registry']:
            sources = {}
            for predicate in criterion['required_predicates']:
                sources[predicate] = sorted({c['source_ref'] for c in trusted['allowed_claims_up_to_event']
                    if c['subject_ref'] == subject and c['predicate'] == predicate})
            receipts[nid][subject] = {'satisfied': bool(sources) and all(sources.values()),
                'public_sources_by_predicate': sources}
    return receipts
