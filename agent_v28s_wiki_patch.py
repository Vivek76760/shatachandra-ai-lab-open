
# ---- v28s: safe wiki_get, no-'query' replies (21 Oct 2026) ----
import json, time
if not globals().get("_wg_safe_ready"):
    _wg_prev = wiki_get
    def wiki_get(params, *a, **k):
        for i in range(3):
            data = _wg_prev(params, *a, **k)
            if params.get("action") != "query" or (isinstance(data, dict) and "query" in data):
                return data
            _wg_cache.pop(json.dumps(params, sort_keys=True), None)
            print(f"[wiki] reply had no 'query', retry {i + 1}", flush=True)
            time.sleep(2)
        raise RuntimeError("Wikipedia kept refusing, last status no query in reply")
    _wg_safe_ready = True
