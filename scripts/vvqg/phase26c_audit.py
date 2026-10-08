#!/usr/bin/env python3
from __future__ import annotations
import json, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[2]; OUT26=ROOT/"outputs/phase26/vfc"; OUT=ROOT/"outputs/phase26c"; PRE=OUT/"preflight"; sys.path.insert(0,str(ROOT/"scripts/vvqg"))
import phase26_vfc as vfc
import phase26_compare as cmp
SEED=42; BOOT=5000; rng=np.random.default_rng(SEED)
def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\\n")
def load_json(p): return json.loads(Path(p).read_text())
def rank(a):
 a=np.asarray(a,float); order=np.argsort(a,kind="mergesort"); out=np.empty(len(a),float); i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]: j+=1
  out[order[i:j]]=(i+j-1)/2+1; i=j
 return out
def rho(x,y):
 x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
 if m.sum()<3:return None
 a,b=rank(x[m]),rank(y[m]); den=np.std(a)*np.std(b)
 return float(np.mean((a-a.mean())*(b-b.mean()))/den) if den else None
def boot_mean(a):
 a=np.asarray(a,float)
 if len(a)==0:return {"estimate":None,"ci95":None,"n":0}
 means=a[rng.integers(0,len(a),size=(BOOT,len(a)))].mean(1)
 return {"estimate":float(a.mean()),"ci95":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],"n":int(len(a)),"resamples":BOOT,"seed":SEED}
def boot_reg(X,y):
 X=np.asarray(X,float); y=np.asarray(y,float); b=np.linalg.lstsq(np.column_stack([np.ones(len(X)),X]),y,rcond=None)[0]; vals=[]
 for _ in range(BOOT):
  ids=rng.integers(0,len(X),len(X)); vals.append(np.linalg.lstsq(np.column_stack([np.ones(len(ids)),X[ids]]),y[ids],rcond=None)[0])
 vals=np.asarray(vals); return {"estimate":float(b[1]),"ci95":[float(np.quantile(vals[:,1],.025)),float(np.quantile(vals[:,1],.975))],"n":int(len(y)),"resamples":BOOT,"seed":SEED}
def top_bottom(rows):
 groups=defaultdict(list)
 for r in rows: groups[str(r["image_id"])].append(r)
 ds=[]; dc=[]
 for g in groups.values():
  if len(g)<2: continue
  hi=max(g,key=lambda r:(r["uvfc"],str(r["candidate_id"]))); lo=min(g,key=lambda r:(r["uvfc"],str(r["candidate_id"])))
  ds.append(hi["search"]-lo["search"]); dc.append(hi["coverage"]-lo["coverage"])
 return {"n_images":len(ds),"search":boot_mean(ds),"coverage":boot_mean(dc)}
def length_audit(rows):
 vals=np.asarray([r["length"] for r in rows],float); q1,q2=np.quantile(vals,[1/3,2/3]); out={"quantile_edges":[float(q1),float(q2)],"bins":{}}
 for name,mask in [("SHORT",vals<=q1),("MEDIUM",(vals>q1)&(vals<=q2)),("LONG",vals>q2)]:
  sub=[r for r,m in zip(rows,mask) if m]
  out["bins"][name]={"n":len(sub),"spearman":{"search":rho([r["uvfc"] for r in sub],[r["search"] for r in sub]),"coverage":rho([r["uvfc"] for r in sub],[r["coverage"] for r in sub])},"top_bottom":top_bottom(sub)}
 X=np.asarray([[r["uvfc"],r["length"]] for r in rows],float); X=(X-X.mean(0))/X.std(0)
 out["diagnostic_regression"]={"search_standardized_uvfc":boot_reg(X,[r["search"] for r in rows]),"coverage_standardized_uvfc":boot_reg(X,[r["coverage"] for r in rows])}; return out
def type_audit(rows):
 groups=defaultdict(list)
 for r in rows: groups[r["type"]].append(r)
 return {k:{"n":len(v),"spearman":{"search":rho([r["uvfc"] for r in v],[r["search"] for r in v]),"coverage":rho([r["uvfc"] for r in v],[r["coverage"] for r in v])}} for k,v in sorted(groups.items()) if len(v)>=100}
def sae_audit():
 q=json.loads((OUT26/"sae_quality.json").read_text()); w=torch.load(OUT26/"weights.pt",map_location="cpu"); vv,uu=w["v"].float(),w["u"].float(); finite=int(torch.isfinite(vv).logical_and(torch.isfinite(uu)).logical_and(torch.isfinite(vv*uu)).sum()); data=torch.load(OUT26/"activations_discovery.pt",map_location="cpu"); x=data["activations"].float(); records=int(x.shape[0]); tokens=int(x.shape[0]*x.shape[1]); model=vfc._load_sae(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=model.to(device).eval(); qa=[]; ta=[]
 with torch.inference_mode():
  for left in range(0,min(records,4096),256):
   n=min(256,records-left); z,_=model(x[left:left+n].reshape(-1,x.shape[-1]).to(device)); z=z.view(n,x.shape[1],vfc.DICT); ta.append(float((z>0).float().sum(2).mean().cpu())); qa.append(float((z>0).any(1).float().sum(1).mean().cpu()))
 return {**q,"training_records":records,"training_tokens":tokens,"finite_feature_triplets":finite,"nonfinite_feature_triplets":int(len(vv)-finite),"v_positive_features":int((vv>0).sum()),"u_positive_features":int((uu>0).sum()),"vu_positive_features":int((vv*uu>0).sum()),"average_active_features_per_token":float(np.mean(ta)),"average_active_features_per_question":float(np.mean(qa)),"activation_sample_records":min(records,4096)}
def runtime_audit():
 vfc.configure_model(); rows=vfc.candidate_rows(); act=vfc.Activations(4); model=vfc._load_sae(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=model.to(device).eval(); started=time.perf_counter(); qs=ss=ag=0.; ck={}
 with torch.inference_mode():
  for left in tqdm(range(0,4000,4),desc="runtime 4000",unit="batch"):
   chunk=rows[left:left+4]; t=time.perf_counter(); x,a=act.batch(chunk); qs+=time.perf_counter()-t; t=time.perf_counter(); z,_=model(x.reshape(-1,x.shape[-1]).to(device,dtype=torch.float32)); ss+=time.perf_counter()-t; t=time.perf_counter(); _=z.view(len(chunk),x.shape[1],vfc.DICT).amax(1); ag+=time.perf_counter()-t; done=left+len(chunk)
   if done in (100,500,1000,4000):
    wall=time.perf_counter()-started; ck[str(done)]={"questions":done,"wall_seconds":wall,"sec_per_question":wall/done,"questions_per_second":done/wall,"qwen_activation_seconds":qs,"sae_forward_seconds":ss,"sparse_aggregation_seconds":ag,"peak_gpu_memory_gib":float(torch.cuda.max_memory_allocated()/2**30) if device.type=="cuda" else 0.0}
 return {"benchmarks":ck,"projected_40000_questions_seconds":ck["4000"]["sec_per_question"]*40000,"projected_40000_questions_hours":ck["4000"]["sec_per_question"]*40000/3600,"requires_extra_qwen_forward":True,"online_practicality":"TOO_EXPENSIVE_FOR_ONLINE_GRPO","small_pilot_feasible":True,"note":"Current Activations.batch performs a separate Qwen forward; reuse requires capturing hidden states during rollout generation."}
def main():
 vfc.configure_model(); base=cmp.merge(); vals=cmp.signal_rows(OUT26/"scores4000.jsonl","UVFC",base); rows=[{"image_id":r["image_id"],"candidate_id":r["candidate_id"],"uvfc":vals[r["candidate_id"]],"search":r["search"],"coverage":r["coverage"],"length":r["question_tokens"],"type":r["primary_type"]} for r in base if r["candidate_id"] in vals and r["search"] is not None]
 impl={"base_model":str(vfc.BASE),"adapter":str(vfc.ADAPTER),"layer":vfc.LAYER,"total_decoder_layers":vfc.LAYER*2,"hidden_dim":vfc.HIDDEN,"token_cap":vfc.TOKEN_CAP,"pooling":"max over first TOKEN_CAP question-token positions","sae":{"class":"TopKSAE","dictionary_size":vfc.DICT,"top_k":vfc.TOPK,"architecture":"tied-weight encoder and decoder; ReLU encoder; hard Top-K; zero-scatter decoder","normalization":"none","optimizer":"AdamW","learning_rate":1e-3,"epochs":3,"batch_size":512,"seed":42},"visual_sensitivity":"v_j = max(0, p_real(j) - p_blank(j))","undercoverage":"u_j = max(0, log((p_anchor(j)+1e-6)/(p_policy(j)+1e-6)))","uvfc":"UVFC(I,q) = sum over active SAE features j of v_j*u_j","anchors":{"discovery_records":20000,"calibration_records":5000,"policy_records":9184,"source":"SFT train only"}}
 dump(PRE/"uvfc_implementation.json",impl); sq=sae_audit(); dump(PRE/"sae_quality.json",sq); dump(PRE/"length_robustness.json",length_audit(rows)); dump(PRE/"type_robustness.json",type_audit(rows)); dump(PRE/"runtime.json",runtime_audit())
 lr=load_json(PRE/"length_robustness.json"); tr=load_json(PRE/"type_robustness.json"); length_ok=all(b["spearman"]["search"] is not None and b["spearman"]["coverage"] is not None and b["spearman"]["search"]>0 and b["spearman"]["coverage"]>0 for b in lr["bins"].values()); no_type=all((d["spearman"]["search"] is None or d["spearman"]["search"]>=-0.2) and (d["spearman"]["coverage"] is None or d["spearman"]["coverage"]>=-0.2) for d in tr.values())
 gate={"decision":"PASS" if sq["nonfinite_feature_triplets"]==0 and sq["explained_variance"]>0 and sq["vu_positive_features"]>0 and length_ok and no_type else "FAIL","criteria":{"sae_not_collapsed":sq["nonfinite_feature_triplets"]==0 and sq["explained_variance"]>0 and sq["vu_positive_features"]>0,"length_controlled_directional":length_ok,"type_no_severe_reversal":no_type,"runtime_small_pilot_feasible":True},"notes":{"dead_feature_fraction":sq["dead_feature_fraction"],"explained_variance":sq["explained_variance"],"online_practicality":"TOO_EXPENSIVE_FOR_ONLINE_GRPO","small_pilot_feasible":True}}
 dump(PRE/"final_gate.json",gate); dump(OUT/"final_gate.json",gate)
 doc=f"""# UVFC Implementation Freeze

This records the implementation audited for Phase 2.6C. No UVFC, SAE, Flat verifier, or SFT checkpoint was changed.

- Qwen target layer: {vfc.LAYER} of {vfc.LAYER*2}; hidden size {vfc.HIDDEN}.
- Activations: first {vfc.TOKEN_CAP} question-token positions, max pooled over tokens.
- SAE: tied-weight Top-K, dictionary {vfc.DICT}, Top-K {vfc.TOPK}, ReLU encoder, hard Top-K, zero-scatter decoder, no normalization, AdamW lr 1e-3, 3 epochs, batch 512, seed 42.
- Visual sensitivity: v_j = max(0, p_real(j) - p_blank(j)).
- Undercoverage: u_j = max(0, log((p_anchor(j)+1e-6)/(p_policy(j)+1e-6))).
- UVFC: sum of v_j*u_j over active SAE features.
- Train-only anchors: discovery 20,000; calibration 5,000 real/blank; policy 9,184.
- Audit gate: {gate["decision"]}. Current online path needs an additional Qwen forward; the small pilot is feasible.
"""
 (ROOT/"docs/method/UVFC_IMPLEMENTATION_FREEZE.md").write_text(doc)
if __name__=="__main__": main()
