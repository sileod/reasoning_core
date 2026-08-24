"""Put warm-protocol (T75/M80) task effects onto the standard-protocol (T300/M20) axis.

The two protocols do NOT produce the same numbers. Measured on the 21 tasks run under both, the warm
protocol applies a per-leg GAIN to margin deltas -- median 1.38x, from 0.39 (winogrande) to 3.13
(themis_py). Ranking is preserved (rho +0.905), which is what the protocol was validated on, but the
LEVELS are not, so warm rows dropped into a standard-protocol matrix look systematically stronger
than they are and any colour scale keyed to absolute effect misreads them.

The fix is an ordinary least-squares calibration per leg, `standard = (warm - intercept) / slope`,
fit on the tasks measured under both protocols. Legs whose fit is weak (|r| < R_FLOOR) get no
calibration -- a slope estimated from a scatter with no trend is worse than leaving the value alone
and saying so.

    from reasoning_core.reports.protocol_calibration import fit, apply_row
    cal = fit(std_rows, warm_rows)        # {leg: {slope, intercept, r, n}}
    apply_row(cal, warm_task_values)      # -> values on the standard axis

Why not use the two anchors every run carries: they are deliberately LOW-EFFECT tasks, so they sit
near the origin where a slope is unidentifiable. They confirm the calibration, they cannot fit it.
"""
from __future__ import annotations

R_FLOOR = 0.5   # below this the per-leg slope is not trustworthy; pass the value through untouched


def fit(std, warm, min_tasks=8):
    """OLS warm-on-standard per leg. `std`/`warm` are {task: {leg: value}}."""
    shared = sorted(set(std) & set(warm))
    legs = sorted({l for t in shared for l in std[t]} & {l for t in shared for l in warm[t]})
    out = {}
    for leg in legs:
        pairs = [(std[t][leg], warm[t][leg]) for t in shared
                 if leg in std[t] and leg in warm[t]]
        if len(pairs) < min_tasks:
            continue
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        sxx = sum((p[0] - mx) ** 2 for p in pairs)
        syy = sum((p[1] - my) ** 2 for p in pairs)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
        if sxx <= 0 or syy <= 0:
            continue
        slope = sxy / sxx
        out[leg] = {"slope": slope, "intercept": my - slope * mx,
                    "r": sxy / ((sxx * syy) ** 0.5), "n": n}
    return out


def apply_row(cal, values):
    """Map one warm-protocol {leg: value} onto the standard axis. Unfittable legs pass through."""
    out = {}
    for leg, v in values.items():
        c = cal.get(leg)
        if c is None or abs(c["r"]) < R_FLOOR or c["slope"] == 0:
            out[leg] = v
        else:
            out[leg] = (v - c["intercept"]) / c["slope"]
    return out


def uncalibrated_legs(cal, legs):
    """Legs that will pass through unchanged -- name them in any figure that shows mixed protocols."""
    return sorted(l for l in legs
                  if l not in cal or abs(cal[l]["r"]) < R_FLOOR or cal[l]["slope"] == 0)
