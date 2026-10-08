import json,time,sys
from pathlib import Path
from collections import Counter
from functools import lru_cache
import torch
from PIL import Image
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from vvqg.paths import local_model_path
from vvqg.rewards.support.frozen_verifier import frozen_prompts,parse_prediction
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
inp=ROOT/'outputs/phase3a_grpo_pilot/preflight/rollouts_sft_4x.jsonl'; out=ROOT/'outputs/phase3a_grpo_pilot/preflight/rollouts_flat.jsonl'
rows=[json.loads(x) for x in inp.read_text().splitlines() if x.strip()]
groups={}
for i,r in enumerate(rows):
 r={**r,'pilot_id':i}; groups.setdefault((str(r['image']),r['question']),[]).append(r)
unique=[v[0] for v in groups.values()]
model_path=local_model_path('Qwen2.5-VL-7B-Instruct'); prompt=frozen_prompts()['flat']; processor=AutoProcessor.from_pretrained(model_path,local_files_only=True); processor.tokenizer.padding_side='left'; model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,device_map='auto',local_files_only=True).eval(); model.requires_grad_(False)
@lru_cache(maxsize=128)
def image(path):
 with Image.open(ROOT/path) as im: return im.convert('RGB')
started=time.perf_counter(); outputs={}
for left in tqdm(range(0,len(unique),32),total=(len(unique)+31)//32,desc='Flat pilot',unit='batch'):
 chunk=unique[left:left+32]; texts=[processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt+'\nQuestion: '+r['question']}]}],tokenize=False,add_generation_prompt=True) for r in chunk]
 inputs=processor(text=texts,images=[image(r['image']) for r in chunk],padding=True,return_tensors='pt').to(model.device)
 with torch.inference_mode(): gen=model.generate(**inputs,max_new_tokens=256,do_sample=False)
 raws=processor.tokenizer.batch_decode(gen[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
 for r,raw in zip(chunk,raws): outputs[(str(r['image']),r['question'])]={**parse_prediction(raw,'flat'),'raw_output':raw}
states=[]
for r in rows:
 parsed=outputs[(str(r['image']),r['question'])]
 states.append({**r,**parsed,'state':{'supported':'NO_CONFIRMED_VIOLATION','unsupported':'CONFIRMED_VIOLATION','uncertain':'UNCERTAIN'}.get(parsed.get('overall_visual_support'),'FAILED') if parsed.get('status')=='ok' else 'FAILED'})
out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in states))
meta={'records':len(states),'unique_questions':len(unique),'wall_seconds':time.perf_counter()-started,'batch_size':32,'counts':dict(Counter(r['state'] for r in states))}
(ROOT/'outputs/phase3a_grpo_pilot/preflight/flat_runtime.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
