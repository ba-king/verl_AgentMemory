"""Read-only conflict checking over explicitly interpreted propositions.

Annotations are AUDIT data, never Packet fields or Teacher messages. A semantic
reviewer identifies proposition dimensions; this core compares exact ownership,
dimension and value. It neither treats shared keywords as identity nor produces
or applies a corrective delta. Unknown prose is UNREVIEWED, never silently PASS.
"""
import copy
import re


def claim(packet,index,dimension,value,*,collection='evidence',need_id=None,subject_ref=None):
    row=packet[collection][index]
    if collection=='evidence':
        need_id,subject_ref=row['need_id'],row['subject_ref']
    if need_id is None or subject_ref is None:raise ValueError('UNRESOLVED_REQUIRES_EXPLICIT_BINDING')
    return {'collection':collection,'index':index,'need_id':need_id,'subject_ref':subject_ref,
            'dimension':dimension,'value':value,'original':copy.deepcopy(row)}


def audit(packet,annotations,trusted_current=()):
    errors=[]
    for c in annotations:
        assert packet[c['collection']][c['index']]==c['original'],'ANNOTATION_DOES_NOT_MATCH_PACKET'
        if c['collection']=='evidence':
            assert (c['need_id'],c['subject_ref'])==(c['original']['need_id'],c['original']['subject_ref'])
    key=lambda c:(c['need_id'],c['subject_ref'],c['dimension'])
    for i,a in enumerate(annotations):
        for b in annotations[i+1:]:
            if key(a)==key(b):
                code='ACTIVE_STATE_CONTRADICTION' if a['value']!=b['value'] else 'DUPLICATE_SUPERSEDED_FACT'
                errors.append({'code':code,'left':[a['collection'],a['index']],
                    'right':[b['collection'],b['index']],'binding':list(key(a))})
        for truth in trusted_current:
            if key(a)==key(truth) and a['value']!=truth['value']:
                errors.append({'code':'STALE_EVIDENCE_NOT_RETRACTED',
                    'active':[a['collection'],a['index']],'binding':list(key(a)),
                    'trusted_source_refs':truth['source_refs']})
    addressed={(c['collection'],c['index']) for c in annotations}
    unreviewed=[[collection,i] for collection in ('evidence','unresolved')
                for i in range(len(packet[collection])) if (collection,i) not in addressed]
    return {'findings':errors,'counts':{code:sum(e['code']==code for e in errors)
                for code in ('STALE_EVIDENCE_NOT_RETRACTED','ACTIVE_STATE_CONTRADICTION','DUPLICATE_SUPERSEDED_FACT')},
            'unreviewed':unreviewed,'scope':'explicitly annotated propositions only',
            'packet_modified':False,'corrective_delta_generated':False}


def literal_annotations(packet):
    """Only unambiguous key=value text, independent of task/domain vocabulary."""
    result=[]
    for i,e in enumerate(packet['evidence']):
        m=re.fullmatch(r'([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*([^=]+)',e['fact'])
        if m:result.append(claim(packet,i,m[1],m[2].strip()))
    return result
