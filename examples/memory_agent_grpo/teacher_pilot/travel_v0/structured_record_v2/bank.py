"""Apply a validated local edit atomically through the existing Bank executor."""
import copy

from ....memory_agent.action.structured_update import apply_record_update, validate_update_arguments
from ..structured_record.bank import StructuredBank as ReplacementBank


class StructuredBank(ReplacementBank):
    def execute(self, action):
        if action.get('action') != 'UPDATE':
            return super().execute(action)
        try:
            args = action['arguments']
            validate_update_arguments(args)
            mid = args['memory_id']
            if mid not in self.visible_ids:
                return {'status': 'REJECTED', 'code': 'ID_NOT_VISIBLE_IN_CURRENT_SESSION'}, None
            if mid not in self.records:
                return {'status': 'REJECTED', 'code': 'MEMORY_ID_NOT_FOUND'}, None
            record = apply_record_update(self.records[mid], args)
        except (ValueError, TypeError, KeyError) as error:
            return {'status': 'REJECTED', 'code': 'INVALID_STRUCTURED_UPDATE', 'detail': str(error)}, None
        # Compute the entire new value first. A failed edit never reaches Store.
        compiled = {'action': 'UPDATE', 'arguments': {'memory_id': mid, 'record': record}}
        start = len(self.actions)
        result, hits = super().execute(compiled)
        if len(self.actions) > start:
            self.actions[-1]['action'] = copy.deepcopy(action)
        return result, hits
