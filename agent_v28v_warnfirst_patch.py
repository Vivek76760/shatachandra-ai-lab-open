
# ---- v28v: put the "runs disagree" warning first (22 Oct 2026) ----
import re
if not globals().get("_warnfirst_ready"):
    _WARN_RE = re.compile(r"\(Warning: repeated runs gave different results")
    _WARN_TOP = "I am NOT sure about this answer: the runs gave different results. Do not trust the first number below.\n"
    def _warn_first(s):
        s = str(s)
        if _WARN_RE.search(s) and not s.startswith(_WARN_TOP[:20]):
            return _WARN_TOP + s
        return s
    _prev_agent_all_wf = agent_all
    def agent_all(q, *a, **k):
        return _warn_first(_prev_agent_all_wf(q, *a, **k))
    _warnfirst_ready = True
