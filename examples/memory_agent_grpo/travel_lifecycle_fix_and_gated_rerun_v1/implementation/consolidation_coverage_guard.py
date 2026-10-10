"""Pure exact structured coverage and one bounded post-YIELD continuation.

No Bank mutation. No automatic ADD, no cross-subject equality shortcut, no gold.
Evidence values are used to verify recovery, not emitted in guard feedback.
"""


def coverage(contract, snapshot):
    checks = []
    for obligation in contract['obligations']:
        proof = contract['public_facts'][obligation['obligation_id']]
        matches, diagnostics = [], []
        for row in snapshot['records']:
            record = row.get('record', {})
            if record.get('subject') != obligation['subject'] or not record.get('scope', '').strip():
                continue
            values = {f['path']:f['value'] for f in record.get('fields', [])}
            missing = [p for p in obligation['required_paths'] if p not in values]
            wrong = [p for p in obligation['required_paths'] if p in values and values[p] != proof['values'][p]]
            diagnostics.append({'memory_id':row['memory_id'],'missing_paths':missing,'inexact_paths':wrong})
            if not missing and not wrong:
                matches.append(row['memory_id'])
        checks.append({'obligation_id':obligation['obligation_id'],'subject':obligation['subject'],
            'covered':bool(matches),'memory_ids':matches,'record_checks':diagnostics,
            'required_paths':obligation['required_paths'],'source_refs':obligation['source_refs']})
    return {'satisfied':all(c['covered'] for c in checks),'checks':checks,
            'code':'COVERED' if all(c['covered'] for c in checks) else 'UNSATISFIED_DURABLE_OBLIGATION'}


class CoverageGuard:
    def __init__(self, contract):
        self.contract = contract
        self.continuations = 0
        self.yield_checks = []

    def check_yield(self, snapshot):
        result = coverage(self.contract, snapshot)
        self.yield_checks.append(result)
        return result

    def allow_continuation(self, snapshot):
        if coverage(self.contract, snapshot)['satisfied'] or self.continuations >= 1:
            return False
        self.continuations += 1
        return True
