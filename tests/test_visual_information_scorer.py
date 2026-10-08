import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import torch
from vvqg.rewards.visual_information import answer_logprobs, cache_key

def test_answer_logprobs_masks_only_right_aligned_answers():
    logits=torch.zeros((2,5,7)); logits[0,3,4]=5; logits[0,4,5]=5; logits[1,3,2]=5
    ids=torch.tensor([[0,1,2,4,5],[0,1,2,2,0]])
    means, token_lp, mask=answer_logprobs(logits,ids,torch.tensor([2,1]))
    assert mask.tolist()==[[True,True],[False,True]]
    assert means.shape==(2,) and torch.isfinite(means).all()

def test_cache_key_includes_condition_image_question_answer_and_model():
    row={'condition':'real','image':'a.jpg','question':'q','answer':'a'}
    base=cache_key(row,'/model'); assert base==cache_key(dict(row),'/model')
    assert base!=cache_key({**row,'condition':'text'},'/model')
    assert base!=cache_key({**row,'answer':'b'},'/model')
