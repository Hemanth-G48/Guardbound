"""Distinguish 'invalid token' (401) from 'valid token, licence not accepted' (403).

Read-only. Prints the account NAME only — never the token value.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

api = HfApi()
out = {}

# 1) Is the stored token valid at all?
try:
    who = api.whoami()
    out["token_valid"] = True
    out["token_type"] = who.get("type")
    out["account_name"] = who.get("name")
    out["is_pro"] = who.get("isPro")
    orgs = who.get("orgs") or []
    out["org_count"] = len(orgs)
    print("token valid            : True")
    print(f"token type             : {out['token_type']}")
    print(f"account                : {out['account_name']}")
    print(f"isPro                  : {out['is_pro']}")
    print(f"orgs                   : {out['org_count']}")
except HfHubHTTPError as e:
    out["token_valid"] = False
    out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(f"token valid            : False -> {out['error']}")

# 2) Which token is being used?
tok_file = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "token"
out["token_file_present"] = tok_file.exists()
if tok_file.exists():
    raw = tok_file.read_text(encoding="utf-8").strip()
    out["token_prefix_class"] = "hf_..." if raw.startswith("hf_") else "non-standard"
    out["token_length"] = len(raw)
print(f"token file present     : {out['token_file_present']}  "
      f"({out.get('token_prefix_class')}, len={out.get('token_length')})")

# 3) Contrast: a gated repo this account CAN read vs the one it cannot.
out["access"] = {}
for repo in ("google/gemma-3-4b-it", "meta-llama/Llama-3.1-8B-Instruct"):
    try:
        api.hf_hub_download(repo, "config.json")
        out["access"][repo] = "GRANTED"
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", "?")
        out["access"][repo] = f"DENIED ({code})"
    except Exception as e:  # noqa: BLE001
        out["access"][repo] = f"{type(e).__name__}"
for k, v in out["access"].items():
    print(f"access {k:<38} {v}")

Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14_4_hf_access_check.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote results/phase14_4_hf_access_check.json")
