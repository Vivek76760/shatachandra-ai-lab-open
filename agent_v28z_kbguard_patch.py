
# ---- v28z: do not save cut-off answers or fallback notes (23 Oct 2026) ----
import re
if not globals().get("_v28z_ready"):
    _kb_good_prev_z = _kb_good
    _FALLBACK_NOTES = ("(A long answer could not be built from Wikipedia", "ध्यान दें: विस्तृत उत्तर नहीं बन सका")
    def _body_cut(text):
        body = re.split(r"\(Source|\(स्रोत|\(మూలం", text)[0].rstrip()
        return len(body.split()) >= 20 and body[-1:] not in ".!?।)\"'”"
    def _kb_good(text):
        if text.lstrip().startswith(_FALLBACK_NOTES):
            return False
        if _body_cut(text):
            return False
        return _kb_good_prev_z(text)
    _v28z_ready = True
