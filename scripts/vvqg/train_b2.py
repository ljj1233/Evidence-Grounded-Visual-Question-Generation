#!/usr/bin/env python3
"""Train B2 with the shared trainer; --max-steps is the mandatory smoke knob."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.baselines.common import read_jsonl
from vvqg.baselines.contrastive_grounding.data import load_embedding_cache, load_image_caches
from vvqg.baselines.contrastive_grounding.model import ContrastiveGroundingQwen
from vvqg.baselines.contrastive_grounding.trainer import B2Dataset, B2Trainer, make_b2_collator
from vvqg.training.trainer_base import load_baseline_model_and_processor, make_training_arguments

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--records',type=Path,default=ROOT/'outputs/same_backbone_baselines/manifests/train50k_records.jsonl'); p.add_argument('--cache',type=Path,required=True); p.add_argument('--model',type=Path,default=Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser()); p.add_argument('--output',type=Path,default=ROOT/'outputs/same_backbone_baselines/b2_contrastive/checkpoints'); p.add_argument('--limit',type=int); p.add_argument('--max-steps',type=int,default=-1); p.add_argument('--device',default='cuda'); p.add_argument('--attn-implementation',default='flash_attention_2'); p.add_argument('--processor-backend',choices=('pil','torchvision'),default='pil'); p.add_argument('--dataloader-num-workers',type=int,default=8); p.add_argument('--per-device-batch-size',type=int,default=2); p.add_argument('--gradient-accumulation-steps',type=int,default=8); p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--min-pixels',type=int,default=4096); p.add_argument('--max-pixels',type=int,default=147456); p.add_argument('--beta-start',type=float,default=10.0); p.add_argument('--beta-step',type=float,default=10.0); args=p.parse_args()
    records=read_jsonl(args.records)
    caches=load_image_caches(args.cache)
    vectors, embedding_dim=load_embedding_cache(args.cache/'embeddings.json')
    if args.limit:
        from vvqg.baselines.contrastive_grounding.data import text_hash
        records = [row for row in records if str(row.get('image_id') or Path(row['image']).stem.rsplit('_', 1)[-1].lstrip('0') or '0') in caches and text_hash(row['question']) in vectors]
        records = records[:args.limit]
    model,processor,_=load_baseline_model_and_processor(args.model, {'enabled':True,'r':16,'lora_alpha':32,'lora_dropout':0.05,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}, args.attn_implementation, args.processor_backend)
    wrapped=ContrastiveGroundingQwen(model, embedding_dim); merge=int(getattr(model.config.vision_config,'spatial_merge_size',2)); dataset=B2Dataset(processor,records,caches,vectors,merge,min_pixels=args.min_pixels,max_pixels=args.max_pixels); collator=make_b2_collator(processor)
    training_args=make_training_arguments(args.output,epochs=2,per_device_batch_size=args.per_device_batch_size,gradient_accumulation_steps=args.gradient_accumulation_steps,gradient_checkpointing=args.gradient_checkpointing,max_steps=args.max_steps,dataloader_num_workers=args.dataloader_num_workers)
    trainer=B2Trainer(model=wrapped,args=training_args,train_dataset=dataset,data_collator=collator,beta_start=args.beta_start,beta_step=args.beta_step); result=trainer.train(); trainer.save_model(); print(json.dumps({'records':len(records),'steps':result.global_step,'output':str(args.output)})); return 0
if __name__=='__main__': raise SystemExit(main())
