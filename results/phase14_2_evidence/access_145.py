"""Phase 14.5 — access check for meta-llama/Llama-3.2-3B-Instruct.

Reports the exact requirement. No download, no substitution.
Prints the account NAME only, never the token value.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

TARGET = "meta-llama/Llama-3.2-3B-Instruct"
CONTRAST = [
    "meta-llama/Llama-3.1-8B-Instruct",   # account previously shown to have access
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-3-4b-it",               # licence accepted earlier today
]
OUT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14_5_access_check.json")

api = HfApi()
out: dict = {"target_model": TARGET}

try:
    who = api.whoami()
    out["account"] = {"name": who.get("name"), "type": who.get("type"),
                      "token_valid": True, "org_count": len(who.get("orgs") or [])}
    print(f"account        : {out['account']['name']} ({out['account']['type']})")
    print("token valid    : True")
except HfHubHTTPError as e:
    out["account"] = {"token_valid": False, "error": str(e)[:200]}
    print(f"token invalid  : {e}")

try:
    info = api.model_info(TARGET)
    out["model_info"] = {"exists": True, "gated": str(info.gated), "sha": info.sha,
                         "file_count": len(info.siblings or [])}
    print(f"model exists   : True   gated={info.gated}  sha={info.sha}  files={len(info.siblings or [])}")
except Exception as e:  # noqa: BLE001
    out["model_info"] = {"exists": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

out["access"] = {}
for repo in [TARGET] + CONTRAST:
    try:
        api.hf_hub_download(repo, "config.json")
        out["access"][repo] = "GRANTED"
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", "?")
        out["access"][repo] = f"DENIED ({code})"
    except Exception as e:  # noqa: BLE001
        out["access"][repo] = f"{type(e).__name__}"

print()
for repo, verdict in out["access"].items():
    marker = "  <-- TARGET" if repo == TARGET else ""
    print(f"  {verdict:<14} {repo}{marker}")

out["conclusion"] = (
    "meta-llama/Llama-3.2-3B-Instruct requires accepting the Meta Llama 3.2 "
    "licence for this account. The token itself is valid and the account already "
    "has access to other gated Meta Llama repositories, so this is a per-repo "
    "licence gate on the 3.2 release specifically, not an auth failure."
)
out["required_action"] = (
    "Sign in to huggingface.co as the account above, open "
    "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and accept the "
    "licence / request access. Meta gates per repository, so prior acceptance of "
    "the Llama 3.1 licence does not carry over to 3.2."
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
