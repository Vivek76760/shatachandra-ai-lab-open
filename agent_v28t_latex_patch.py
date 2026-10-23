
# ---- v28t: remove LaTeX $...$ signs from numbers (22 Oct 2026) ----
import re
if not globals().get("_latex_ready"):
    _MATH = re.compile(r"\$([^$\n]{1,80})\$")
    def _clean_math(s):
        def fix(m):
            inner = m.group(1)
            probe = re.sub(r"\\text\{[^}]*\}", "", inner)
            probe = re.sub(r"\\[A-Za-z]+", "", probe)
            if re.search(r"[A-Za-z]", probe) or not re.search(r"[\d\\]", inner):
                return m.group(0)
            x = re.sub(r"\\text\{([^}]*)\}", r"\1", inner)
            x = x.replace(r"\times", "×").replace(r"\approx", "≈")
            x = x.replace(r"\,", " ").replace(r"\ ", " ")
            x = re.sub(r"\^\{([^}]*)\}", r"^\1", x)
            return x.strip()
        return _MATH.sub(fix, s)
    _prev_agent_all_ltx = agent_all
    def agent_all(q, *a, **k):
        return _clean_math(str(_prev_agent_all_ltx(q, *a, **k)))
    _latex_ready = True
