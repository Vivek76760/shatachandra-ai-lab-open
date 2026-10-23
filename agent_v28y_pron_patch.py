
# ---- v28y: remove leftover pronunciation junk (23 Oct 2026) ----
import re
if not globals().get("_v28y_ready"):
    _PR_EMPTY = re.compile(r"\b[A-Z][a-z]+ pronunciation:\s*;\s*")
    _PR_IPA = re.compile(r"\(\s*[^()]*?/[^()]*?/\s*;\s*pronunciation;\s*")
    _PR_BARE = re.compile(r"\s*;\s*pronunciation\s*(?=[;)])")
    _PR_OPEN = re.compile(r"\(\s*pronunciation\s*[;:]\s*")
    def _clean_pron(s):
        s = _PR_IPA.sub("(", s)
        s = _PR_EMPTY.sub("", s)
        s = _PR_OPEN.sub("(", s)
        s = _PR_BARE.sub("", s)
        return s
    _prev_agent_all_pron = agent_all
    def agent_all(q, *a, **k):
        return _clean_pron(str(_prev_agent_all_pron(q, *a, **k)))
    _v28y_ready = True
