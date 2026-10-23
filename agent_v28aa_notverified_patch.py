# ---- v28aa: "not fully verified" line first when an answer carries a weak-source note (23 Oct 2026) ----
import re
if not globals().get("_v28aa_ready"):
    _NV_WEAK = re.compile(r"does not explicitly say|Only one source found|No source checked")
    _NV_TEXT = {
        "English": "Not fully verified: please check this answer yourself before relying on it.\n",
        "Hindi": "पूरी तरह जाँचा नहीं गया: कृपया इस उत्तर पर भरोसा करने से पहले स्वयं जाँच लें।\n",
    }
    def _nv_apply(question, text):
        text = str(text)
        if not _NV_WEAK.search(text):
            return text
        try:
            lang = detect_lang(question)
        except Exception:
            lang = "English"
        line = _NV_TEXT.get(lang, _NV_TEXT["English"])
        if text.startswith(line):
            return text
        return line + text
    _prev_agent_all_nv = agent_all
    def agent_all(q, *a, **k):
        return _nv_apply(q, _prev_agent_all_nv(q, *a, **k))
    _v28aa_ready = True
