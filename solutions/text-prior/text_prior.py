import json
import math
import os

_ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_logreg.json")
_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        with open(_ARTIFACT, "r", encoding="utf-8") as fh:
            _MODEL = json.load(fh)
    return _MODEL


def n_words(problem):
    return len(str(problem).split())


def scores(problems):
    m = _load()
    out = []
    for problem in problems:
        z = (math.log1p(n_words(problem)) - m["mean"]) / m["scale"]
        out.append(1.0 / (1.0 + math.exp(-(m["coef"] * z + m["intercept"]))))
    return out


def predict(problems, threshold=0.5):
    return [bool(s > threshold) for s in scores(problems)]
