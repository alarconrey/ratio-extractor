"""Section rule, fixed before training: every paragraph of the legal-grounds section is the ratio."""

import numpy as np


def rule_predict(frame):
    return (frame.seccion == "fundamentos").to_numpy()


def rule_probabilities(frame):
    """0/1 scores so the rule can be served through the same interface as the models."""
    return rule_predict(frame).astype(float)


def has_foundations_section(frame):
    return bool(np.any(frame.seccion == "fundamentos"))
