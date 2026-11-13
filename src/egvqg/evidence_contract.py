"""Audited representation changes; these functions never certify semantic truth."""
from __future__ import annotations

import copy

from .metrics import validate_necessity_item
from .records import canonical_json, sha256_text


def normalize_v1(output, removed, knowledge, component_ids):
    result=copy.deepcopy(output);changes=[]
    if result.get('removed')=='T或I':
        changes.append({'field':'removed','before':'T或I','after':removed,'rule':'literal_placeholder'})
        result['removed']=removed
    for cid,item in result.get('components',{}).items():
        if not isinstance(item,dict):continue
        for i,premise in enumerate((item.get('path') or {}).get('premises',[])):
            source=premise.get('source')
            if source in ('K1','K2','K3','K4') and source in knowledge:
                premise.update(source='K',knowledge_item=source,source_raw=source)
                changes.append({'field':f'components/{cid}/path/premises/{i}/source',
                                'before':source,'after':'K','rule':'configured_knowledge_alias'})
        candidate={**item,'path':None}
        if item.get('proposed')==1 and item.get('path') and not validate_necessity_item(candidate,removed,component_ids):
            witness=item['witness']
            if witness['completion_a']['values'][cid]!=witness['completion_b']['values'][cid]:
                extra=item['path'];item['path']=None
                item.setdefault('annotations',{}).update(extra_path=extra,requires_modality_review=True)
                changes.append({'field':f'components/{cid}/path','before':extra,
                    'after':'annotations/extra_path','rule':'retain_extra_path_for_semantic_review'})
        item.setdefault('annotations',{})['requires_modality_review']=True
    return result,changes


def decode_v3(raw, removed, reference, material, knowledge):
    """Map a single evidence object to the existing decision representation.

    A is bound to actual reviewed input, not re-described by the model. Raw output
    remains in calls.jsonl and in the analysis round. No repair is applied here.
    """
    result={'removed':raw.get('removed',removed),'components':{},'decisive_facts':raw.get('decisive_facts',[])}
    actual_values={c['id']:c['value'] for c in reference['components']}
    components=raw.get('components',{})
    if not isinstance(components,dict):return result,{'contract':'v3','error':'components_not_object'}
    for cid,entry in components.items():
        if not isinstance(entry,dict):result['components'][cid]=None;continue
        kind=entry.get('kind');data=entry.get('evidence') or {};errors=[]
        if not isinstance(data,dict):data={};errors.append('evidence_not_object')
        item={'proposed':None,'path':None,'witness':None,'unresolved_reason':None,
              'annotations':{'note':entry.get('note'),'requires_modality_review':True}}
        if kind=='path':
            item.update(proposed=0,path=data)
            for premise in data.get('premises',[]):
                if premise.get('source')=='K' and premise.get('knowledge_item') not in {
                        k for k in knowledge if k in ('K1','K2','K3','K4')}:
                    errors.append('missing_knowledge_item')
        elif kind=='witness':
            item['proposed']=1
            edits=data.get('edits')
            if not isinstance(edits,list) or not edits or any(not isinstance(e,dict) or not all(e.get(k) is not None for k in ('locator','from','to'))
                                or e.get('from')==e.get('to') for e in edits):errors.append('invalid_edits')
            actual=material['text'] if removed=='T' else 'Original image SHA256: '+material['image_sha256']
            item['witness']={'completion_a':{'content':actual,'compatible_because':'Actual reviewed input',
                    'values':actual_values,'actual_material_sha256':sha256_text(material['text']) if removed=='T'
                          else material['image_sha256']},
                'completion_b':{'content':data.get('alternative_content'),
                    'compatible_because':data.get('compatible_because'),
                    'values':data.get('alternative_values'),'edits':edits}}
            if 'actual_values' in data and data['actual_values']!=actual_values:errors.append('actual_value_override')
        elif kind=='unresolved':item['unresolved_reason']=data.get('reason') or entry.get('note')
        else:errors.append('invalid_evidence_kind');item['unresolved_reason']='Invalid evidence kind'
        if 'proposed' in entry or 'path' in entry or 'witness' in entry:
            errors.append('mixed_contract_fields')
        if errors:item['contract_errors']=errors
        result['components'][cid]=item
    return result,{'contract':'v3','raw_sha256':sha256_text(canonical_json(raw)),
                  'actual_values_bound_to_reviewed_reference':True}
