import math
import os
import tempfile
import subprocess
import json
import udocker

def score_scalar(answer, entry, k=10.0):
    import math
    """
    Scores an answer based on a unified, scaled error metric.
    This version uses a steep penalty (k=10.0) for high sensitivity.
    """
    reference = float(entry.answer)
    try:
        submitted = float(str(answer).split('=')[-1].strip().rstrip('.'))
    except (ValueError, TypeError):
        return 0.0

    # Unified error: abs_err / (abs_ref + 1).
    normalized_error = abs(submitted - reference) / (abs(reference) + 1.0)

    # Exponential decay with a very strict penalty (k=10.0)
    semantic_reward = math.exp(-k * normalized_error)

    try:
        float(str(answer))
        format_reward = 1.0
    except ValueError:
        format_reward = 0.75

    return semantic_reward * format_reward


