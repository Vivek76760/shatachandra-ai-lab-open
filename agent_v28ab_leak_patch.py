# ---- v28ab: remove leaked tool text from answers (23 Oct 2026) ----
import re
if not globals().get("_v28ab_ready"):
    _LEAK_LINE = re.compile(r"TOOL:\s*lookup\([^\n]*", re.I)
    _LEAK_SENT = re.compile(r"[^.\n]*\b(?:use|using)\s+the\s+(?:provided\s+)?(?:lookup\s+)?tool[^.\n]*\.?", re.I)
    _NV_LINES = re.compile(r"^(Not fully verified|पूरी तरह जाँचा नहीं गया).*$", re.M)
    def _strip_leak(question, text):
        text = str(text)
        if not _LEAK_LINE.search(text):
            return text
        t = _LEAK_LINE.sub("", text)
        t = _LEAK_SENT.sub("", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        body = re.sub(r"\(.*?\)", "", _NV_LINES.sub("", t), flags=re.S)
        if len(body.split()) < 5:
            try:
                lang = detect_lang(question)
            except Exception:
                lang = "English"
            return NOT_FOUND.get(lang, NOT_FOUND["English"])
        return t
    _prev_agent_all_ab = agent_all
    def agent_all(q, *a, **k):
        return _strip_leak(q, _prev_agent_all_ab(q, *a, **k))
    _v28ab_ready = True
