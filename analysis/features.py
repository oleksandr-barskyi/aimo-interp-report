import re, numpy as np, pandas as pd

GEO = r"triangle|circle|angle|square|rectangle|polygon|circumcircle|radius|perpendicular|parallel|vertex|vertices|diagonal|hexagon|area|perimeter"
NT  = r"prime|divisib|integer|remainder|modulo|\\bmod\\b|digit|factorial|gcd|lcm|coprime|multiple of"
CMB = r"probability|ways|choose|permutation|arrangement|distinct|subset|sequence of|randomly|selected"
ALG = r"polynomial|function|equation|root|coefficient|real number|complex|matrix|logarithm|sum of|product of"

def feats(texts):
    rows = []
    for t in texts:
        t = str(t)
        words = t.split()
        nums = re.findall(r"\d+", t)
        vals = [int(n) for n in nums if len(n) < 12] or [0]
        low = t.lower()
        rows.append(dict(
            n_chars=len(t),
            n_words=len(words),
            mean_word_len=float(np.mean([len(w) for w in words])) if words else 0.0,
            n_sent=t.count(".") + t.count("?") + t.count("!"),
            n_qmark=t.count("?"),
            n_digits=sum(c.isdigit() for c in t),
            n_nums=len(nums),
            log_max_num=float(np.log1p(max(vals))),
            log_mean_num=float(np.log1p(np.mean(vals))),
            n_dollar=t.count("$"),
            n_bslash=t.count("\\"),
            n_frac=low.count("\\frac"),
            n_sqrt=low.count("\\sqrt"),
            n_caret=t.count("^"),
            n_underscore=t.count("_"),
            n_comma=t.count(","),
            latex_ratio=t.count("$") / max(len(words), 1),
            n_upper_vars=len(re.findall(r"\b[A-Z]\b", t)),
            cap_ratio=sum(c.isupper() for c in t) / max(len(t), 1),
            geo=len(re.findall(GEO, low)),
            nt=len(re.findall(NT, low)),
            cmb=len(re.findall(CMB, low)),
            alg=len(re.findall(ALG, low)),
            starts_find=int(low.startswith("find")),
            starts_what=int(low.startswith("what")),
            starts_how=int(low.startswith("how many")),
            starts_let=int(low.startswith("let")),
            has_compute=int("compute" in low),
            has_prove=int("prove" in low or "show that" in low),
            has_least=int("least" in low or "smallest" in low or "minimum" in low),
            has_largest=int("largest" in low or "greatest" in low or "maximum" in low),
            has_all=int("all " in low),
            has_positive_int=int("positive integer" in low),
            has_frac_answer=int("m+n" in low.replace(" ", "") or "relatively prime" in low or "coprime" in low),
        ))
    return pd.DataFrame(rows)
