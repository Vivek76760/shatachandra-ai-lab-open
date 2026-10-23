
# ---- v28w: Hindi digit-table clash + long-answer request fixes (23 Oct 2026) ----
import re
if not globals().get("_v28w_ready"):
    _HI_DIGITS = {2406: 48, 2407: 49, 2408: 50, 2409: 51, 2410: 52, 2411: 53, 2412: 54, 2413: 55, 2414: 56, 2415: 57}
    _hi_request_prev_w = hi_request
    def hi_request(q):
        global _DEV
        saved = _DEV
        _DEV = _HI_DIGITS
        try:
            return _hi_request_prev_w(q)
        finally:
            _DEV = saved
    _la_request_prev_w = la_request
    _LA_CALC = re.compile(r"^\s*(calculate|compute|solve|convert)\b", re.I)
    _LA_LONGART = re.compile(r"\b(long|detailed|full|complete)\s+(article|essay|write-?up|passage|piece)\b", re.I)
    _LA_TOPIC_AFTER = re.compile(r"\b(?:about|on|of)\s+(?:the\s+)?(.+?)\s*[?.!]*$", re.I)
    _LA_JUNK = re.compile(r"^(?:answer|i need|i want|please)\s+", re.I)
    def la_request(q):
        if _LA_CALC.match(q):
            return None
        r = _la_request_prev_w(q)
        if r is None and _LA_LONGART.search(q):
            m = _LA_TOPIC_AFTER.search(q)
            if m and 2 <= len(m.group(1)) <= 60 and len(m.group(1).split()) <= 8:
                return (250, m.group(1).strip())
            return None
        if r:
            words, t = r
            for _ in range(3):
                t2 = _LA_JUNK.sub("", t.strip())
                if t2 == t:
                    break
                t = t2
            return (words, t) if len(t) >= 2 else None
        return None
    _v28w_ready = True
