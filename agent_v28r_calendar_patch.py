
# ---- v28r: calendar guard (21 Oct 2026) ----
import re
if not globals().get("_cal_guard_ready"):
    _old_needs_outside = _needs_outside
    _CALCUE = re.compile(r"\b(days?|weeks?|date)\b.*\b(left|after|between|until|before|remaining)\b|\b(left|after|between|until)\b.*\b(days?|weeks?|date)\b|\bday of the week\b", re.I)
    def _needs_outside(t):
        if _CALCUE.search(t):
            t = re.sub(r"\bYYYY(-MM-DD)?\b", " ", t)
            t = re.sub(r"\bin (19|20)\d\d\b", " ", t)
        return _old_needs_outside(t)
    _cal_guard_ready = True
