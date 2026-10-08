"""Visual-information diagnostics; no reward aggregation."""
import re

import numpy as np

# GT-Vision-Lab/VQA PythonEvaluationTools/vqaEvaluation/vqaEval.py
CONTRACTIONS = {'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've", 'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've", 'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't", 'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've", 'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's", "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've", 'isnt': "isn't", 'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've", 'itll': "it'll", "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", 'mightve': "might've", 'mustnt': "mustn't", 'mustve': "must've", 'neednt': "needn't", 'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", 'shant': "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", 'shouldve': "should've", 'shouldnt': "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", 'somebodyll': "somebody'll", 'somebodys': "somebody's", 'someoned': "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", 'someonell': "someone'll", 'someones': "someone's", 'somethingd': "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", 'somethingll': "something'll", 'thats': "that's", 'thered': "there'd", "thered've": "there'd've", "there'dve": "there'd've", 'therere': "there're", 'theres': "there's", 'theyd': "they'd", "theyd've": "they'd've", "they'dve": "they'd've", 'theyll': "they'll", 'theyre': "they're", 'theyve': "they've", 'twas': "'twas", 'wasnt': "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 'weve': "we've", 'werent': "weren't", 'whatll': "what'll", 'whatre': "what're", 'whats': "what's", 'whatve': "what've", 'whens': "when's", 'whered': "where'd", 'wheres': "where's", 'whereve': "where've", 'whod': "who'd", "whod've": "who'd've", "who'dve": "who'd've", 'wholl': "who'll", 'whos': "who's", 'whove': "who've", 'whyll': "why'll", 'whyre': "why're", 'whys': "why's", 'wont': "won't", 'wouldve': "would've", 'wouldnt': "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", 'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", 'youd': "you'd", "youd've": "you'd've", "you'dve": "you'd've", 'youll': "you'll", 'youre': "you're", 'youve': "you've"}
NUMBER_WORDS = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

def normalize_answer(answer):
    original = answer.replace("\n", " ").replace("\t", " ").strip()
    text = original
    numeric_comma = re.search(r"(\d)(,)(\d)", original)
    for p in PUNCT:
        text = text.replace(p, "" if p + " " in original or " " + p in original or numeric_comma else " ")
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    words = (NUMBER_WORDS.get(w, w) for w in text.lower().split())
    return " ".join(CONTRACTIONS.get(w, w) for w in words if w not in {"a", "an", "the"})


def ranks(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inv]


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def summary(x):
    x = np.asarray(x)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "positive_rate": float((x > 0).mean()),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}


def stage_a_statistics(m, gt, gb):
    return {"switch_accuracy": float((m > 0).mean()),
            "switch_mean_margin": float(m.mean()), "switch_median_margin": float(np.median(m)),
            "g_text_mean": float(gt.mean()), "g_text_median": float(np.median(gt)),
            "g_blank_mean": float(gb.mean()), "g_blank_median": float(np.median(gb)),
            "positive_g_text_rate": float((gt > 0).mean()), "positive_g_blank_rate": float((gb > 0).mean()),
            "spearman": spearman(gt.ravel(), gb.ravel()),
            "sign_agreement": float((np.sign(gt) == np.sign(gb)).mean())}


def pair_bootstrap(m, gt, gb, n=5000):
    rng = np.random.default_rng(42)
    results = []
    for _ in range(n):
        idx = rng.integers(len(m), size=len(m))
        results.append(stage_a_statistics(m[idx], gt[idx], gb[idx]))
    return {k: [r[k] for r in results] for k in results[0]}


def auc(negative, positive):
    n, p = len(negative), len(positive)
    rank = ranks(np.concatenate([negative, positive])) + 1
    return float((rank[n:].sum() - p * (p + 1) / 2) / (n * p))


def construct_statistics(negative, positive, n=5000):
    negative, positive = np.asarray(negative), np.asarray(positive)
    if not len(negative) or not len(positive):
        return {"available": False, "n_prior": len(negative), "n_required": len(positive)}
    rng = np.random.default_rng(42)
    boot = np.empty((n, 3))
    for i in range(n):
        a = negative[rng.integers(len(negative), size=len(negative))]
        b = positive[rng.integers(len(positive), size=len(positive))]
        boot[i] = b.mean() - a.mean(), np.median(b) - np.median(a), auc(a, b)
    estimates = positive.mean() - negative.mean(), np.median(positive) - np.median(negative), auc(negative, positive)
    return {"available": True, "n_prior": len(negative), "n_required": len(positive),
            **{k: {"estimate": float(v), "ci95": np.percentile(boot[:, i], [2.5, 97.5]).tolist()}
               for i, (k, v) in enumerate(zip(["mean_difference", "median_difference", "auroc"], estimates))}}
