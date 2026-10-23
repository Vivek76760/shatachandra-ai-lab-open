
# ---- v28u: capitalised maths words in the outside-facts guard (22 Oct 2026) ----
import re
if not globals().get("_caps3_ready"):
    _old_needs_outside_u3 = _needs_outside
    _MATHNAMES = re.compile(r"\b(Fibonacci|Euclid|Euclidean|Collatz|Armstrong|Catalan|Pythagorean|Pythagoras)\b")
    _CAPS = re.compile(r"\b[A-Z]{2,}\b")
    _ONECAP = re.compile(r"\b[A-Z]\b")
    _FACTSTART = re.compile(r"^\s*(who|when|where|which|why)\b", re.I)
    def _needs_outside(t):
        t2 = t
        if re.search(r"\d", t2) and not _FACTSTART.match(t2):
            t2 = _MATHNAMES.sub(" ", t2)
            t2 = _ONECAP.sub(lambda m: m.group(0).lower(), t2)
        if _CUES.search(t2):
            t2 = _CAPS.sub(lambda m: m.group(0).lower(), t2)
        return _old_needs_outside_u3(t2)
    _caps3_ready = True
