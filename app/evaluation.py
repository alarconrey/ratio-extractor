"""Metrics computed per resolution, plus intervals and paired comparisons."""

import math

import numpy as np

from app.settings import BOOTSTRAP_SAMPLES, GOOD_EXTRACTION_F1, SEED


def resolution_scores(frame, predicted):
    """Word-weighted F1 of the extracted paragraphs against the gold ratio, one value per resolution.

    Returns a pandas Series indexed by id_resolucion. A resolution with no predicted and no gold
    words cannot occur here, because every evaluated resolution has at least one gold paragraph.
    """
    data = frame.assign(predicted=np.asarray(predicted, dtype=bool), gold=frame.es_ratio.astype(bool))
    data = data.assign(tp=data.n_palabras * (data.predicted & data.gold), pw=data.n_palabras * data.predicted, gw=data.n_palabras * data.gold)
    totals = data.groupby("id_resolucion")[["tp", "pw", "gw"]].sum()
    return (2 * totals.tp / (totals.pw + totals.gw)).where(totals.tp > 0, 0.0)


def paragraph_scores(gold, predicted):
    gold, predicted = np.asarray(gold, dtype=bool), np.asarray(predicted, dtype=bool)
    true_positive = int((gold & predicted).sum())
    precision = true_positive / max(int(predicted.sum()), 1)
    recall = true_positive / max(int(gold.sum()), 1)
    f1 = 0.0 if true_positive == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def choose_threshold(frame, probabilities, candidates):
    """Pick the probability threshold with the best mean resolution F1 (validation only)."""
    scores = {float(threshold): float(resolution_scores(frame, probabilities >= threshold).mean()) for threshold in candidates}
    best = max(scores, key=lambda threshold: (scores[threshold], -abs(threshold - 0.5)))
    return best, scores


def bootstrap_interval(values, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    """95% percentile interval of the mean, resampling resolutions."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return [0.0, 0.0]
    share = successes / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def mcnemar_exact(first_success, second_success):
    """Exact two-sided McNemar test on paired binary outcomes (same resolutions)."""
    first, second = np.asarray(first_success, dtype=bool), np.asarray(second_success, dtype=bool)
    only_first, only_second = int((first & ~second).sum()), int((~first & second).sum())
    discordant = only_first + only_second
    if discordant == 0:
        return {"only_first": 0, "only_second": 0, "p_value": 1.0}
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1)) / 2 ** discordant
    return {"only_first": only_first, "only_second": only_second, "p_value": min(1.0, 2 * tail)}


def summarize(frame, predicted):
    """All reported figures for one method on one partition."""
    scores = resolution_scores(frame, predicted)
    good = int((scores >= GOOD_EXTRACTION_F1).sum())
    result = {
        "resolution_f1": float(scores.mean()), "resolution_f1_ci95": bootstrap_interval(scores.values),
        "resolution_f1_median": float(scores.median()),
        "share_good": good / len(scores), "share_good_ci95": wilson_interval(good, len(scores)),
        "share_below_0_3": float((scores < 0.3).mean()), "resolutions": int(len(scores)),
        "paragraph": paragraph_scores(frame.es_ratio, predicted),
    }
    meta = frame.drop_duplicates("id_resolucion").set_index("id_resolucion")
    for column, key in [("tipo_nombre", "by_type"), ("analista", "by_analyst")]:
        groups = meta[column].reindex(scores.index)
        result[key] = [
            {"group": str(value), "resolutions": int(mask.sum()), "resolution_f1": float(scores[mask].mean())}
            for value, mask in ((value, groups == value) for value in groups.dropna().unique())
        ]
        result[key].sort(key=lambda row: -row["resolutions"])
    return result, scores
