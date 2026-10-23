
# ---- v28x: Hindi page must cover the whole topic (23 Oct 2026) ----
import re
if not globals().get("_v28x_ready"):
    _HI_PARTICLES = {"का", "के", "की", "में", "पर", "से", "और", "एवं"}
    def _hi_toks(s):
        return [w for w in re.split(r"[\s\-\.॰·,()]+", s) if w and w not in _HI_PARTICLES]
    _hi_title_prev_x = hi_title
    def hi_title(topic):
        t = _hi_title_prev_x(topic)
        want = set(_hi_toks(topic))
        if t and want and not want <= set(_hi_toks(t)):
            try:
                j = _on_hi(wiki_get, {"action": "query", "format": "json", "list": "search",
                                      "srsearch": topic, "srlimit": 8})
                cands = [r["title"] for r in j.get("query", {}).get("search", [])
                         if want <= set(_hi_toks(r["title"]))]
                if cands:
                    return min(cands, key=lambda x: len(_hi_toks(x)))
            except Exception:
                pass
        return t
    _v28x_ready = True
