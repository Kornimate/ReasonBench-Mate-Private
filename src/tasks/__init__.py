def _safe_import(module_name: str):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=["*"])
        exported = getattr(module, "__all__", None)
        if exported is None:
            exported = [name for name in dir(module) if not name.startswith("_")]
        globals().update({name: getattr(module, name) for name in exported})
    except Exception:
        # Some task packages require optional third-party dependencies or
        # environment-specific API keys at import time. Keep the task registry
        # robust so unrelated tasks remain runnable.
        return


_safe_import("game24")
_safe_import("hle")
_safe_import("hotpotqa")
_safe_import("humaneval")
_safe_import("scibench")
_safe_import("mtsamples")
_safe_import("mimic_rrs")
_safe_import("pubmed_qa")
_safe_import("sonnetwriting")
