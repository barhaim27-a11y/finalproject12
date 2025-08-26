import sys, os, io, traceback, importlib.util
import streamlit as st

st.set_page_config(page_title="Parkinsons – Diagnostic Launcher", layout="wide")
st.title("🩺 Diagnostic Preflight for Imports")

def compile_check(py_path: str):
    try:
        # Read as UTF-8-sig to avoid BOM issues
        with open(py_path, "r", encoding="utf-8-sig") as f:
            src = f.read()
        compile(src, py_path, "exec")
        return True, None
    except SyntaxError as e:
        # Build a friendly error block with context
        ctx = ""
        try:
            with open(py_path, "r", encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
            start = max(e.lineno - 3, 0)
            end = min(e.lineno + 2, len(lines))
            ctx = "\n".join(f"{i+1:04d}: {lines[i]}" for i in range(start, end))
        except Exception:
            pass
        details = f"File: {py_path}\nLine: {getattr(e, 'lineno', '?')}  Offset: {getattr(e, 'offset', '?')}\nMsg: {e.msg}\n\nContext:\n{ctx}"
        return False, details
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def safe_import(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module

repo_dir = os.path.dirname(__file__)
cfg_path = os.path.join(repo_dir, "config.py")
pipe_path = os.path.join(repo_dir, "model_pipeline.py")

st.write("Python:", sys.version)

ok1, err1 = compile_check(cfg_path)
ok2, err2 = compile_check(pipe_path)

col1, col2 = st.columns(2)
with col1:
    st.subheader("config.py")
    if ok1:
        st.success("✅ Syntax OK")
    else:
        st.error("❌ SyntaxError in config.py")
        st.code(err1 or "", language="text")
with col2:
    st.subheader("model_pipeline.py")
    if ok2:
        st.success("✅ Syntax OK")
    else:
        st.error("❌ SyntaxError in model_pipeline.py")
        st.code(err2 or "", language="text")

if not (ok1 and ok2):
    st.stop()

# If both valid — proceed to run the main app logic (import and continue)
config = safe_import("config", cfg_path)
mp = safe_import("model_pipeline", pipe_path)

st.success("Imports succeeded — you can now replace this file with your main app or continue to app ▶️")

# --- Minimal app to verify runtime ---
import pandas as pd
df = mp.load_data(config.TRAIN_DATA_PATH)
st.write("Data shape:", df.shape)
st.dataframe(df.head(10), use_container_width=True)

if st.button("Evaluate Production Model (if exists)"):
    try:
        if mp.has_production():
            ev = mp.evaluate_model(config.MODEL_PATH, artifact_tag="diag_best")
            st.json(ev["metrics"])
        else:
            st.warning("No production model found.")
    except Exception as e:
        st.exception(e)
