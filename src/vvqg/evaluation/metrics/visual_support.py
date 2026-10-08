from __future__ import annotations
import random
from collections import Counter

def classification_metrics(y_true, y_pred, labels):
    cm = [[0 for _ in labels] for _ in labels]; index = {x:i for i,x in enumerate(labels)}
    for t,p in zip(y_true,y_pred): cm[index[t]][index[p]] += 1
    n = len(y_true); acc = sum(cm[i][i] for i in range(len(labels))) / n if n else 0.0
    per = {}
    recalls=[]; f1s=[]
    for i, label in enumerate(labels):
        tp=cm[i][i]; fn=sum(cm[i])-tp; fp=sum(row[i] for row in cm)-tp
        precision=tp/(tp+fp) if tp+fp else None; recall=tp/(tp+fn) if tp+fn else None
        f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.0
        per[label]={"precision":precision,"recall":recall,"f1":f1}
        if recall is not None: recalls.append(recall)
        f1s.append(f1)
    return {"n":n,"accuracy":acc,"balanced_accuracy":sum(recalls)/len(recalls) if recalls else 0.0,
            "macro_f1":sum(f1s)/len(labels),"per_class":per,"confusion_matrix":cm}

def paired_comparison(y_true, left, right, supported_index, resamples=1000, seed=42):
    rng=random.Random(seed); out={}
    for name, fn in (("accuracy", lambda y,p: sum(a==b for a,b in zip(y,p))/len(y)),):
        diffs=[]; n=len(y_true)
        for _ in range(resamples):
            idx=[rng.randrange(n) for _ in range(n)]; diffs.append(fn([y_true[i] for i in idx],[left[i] for i in idx])-fn([y_true[i] for i in idx],[right[i] for i in idx]))
        out[name]={"difference":fn(y_true,left)-fn(y_true,right),"ci95":[sorted(diffs)[int(.025*resamples)],sorted(diffs)[int(.975*resamples)]]}
    return out
