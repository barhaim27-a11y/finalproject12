# (same Streamlit app as earlier; kept concise here for packaging)
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config, model_pipeline as mp

st.set_page_config(page_title="Parkinsons – Pro (v7)", layout="wide")
st.title("🧪 Parkinsons – ML App (Pro, v7)")
st.caption("Data & EDA • Single • Multi • Best • Predict • Retrain")

@st.cache_data
def load_df():
    return mp.load_data(config.TRAIN_DATA_PATH)

def read_csv_flex(file) -> pd.DataFrame:
    for enc in ["utf-8","latin-1","cp1255"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception: continue
    file.seek(0); return pd.read_csv(file, errors="ignore")

df = load_df()
features = config.FEATURES; target = config.TARGET
tab_data, tab_single, tab_multi, tab_best, tab_predict, tab_retrain = st.tabs(["📊 Data & EDA","🎯 Single Model","🏁 Multi Compare","🏆 Best Dashboard","🔮 Predict","🔁 Retrain"])

with tab_data:
    st.subheader("Dataset")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(30), use_container_width=True)
    with st.expander("🔎 EDA (expand)", expanded=False):
        left, right = st.columns([1.2,1])
        with left:
            miss_df = df[features + [target]].isna().sum().sort_values(ascending=False).head(20).rename("missing").reset_index().rename(columns={"index":"column"})
            st.dataframe(miss_df, use_container_width=True)
            st.download_button("⬇️ missing.csv", miss_df.to_csv(index=False), "missing.csv", "text/csv")
            desc_df = df[features].describe().T
            st.dataframe(desc_df, use_container_width=True)
            st.download_button("⬇️ describe.csv", desc_df.to_csv(), "describe.csv", "text/csv")
        with right:
            cls = df[target].value_counts().rename({0:"No-PD", 1:"PD"})
            st.bar_chart(cls)
        corr = df[features + [target]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

def edit_params(model_name: str, key_prefix: str=""):
    import config
    params = config.DEFAULT_PARAMS.get(model_name, {}).copy()
    cols = st.columns(3); edited={}; i=0
    for k,v in params.items():
        with cols[i%3]:
            skey=f"{key_prefix}{model_name}_{k}"
            if isinstance(v,bool): edited[k]=st.checkbox(k,value=v,key=skey)
            elif isinstance(v,int): edited[k]=st.number_input(k,value=int(v),step=1,key=skey)
            elif isinstance(v,float): edited[k]=st.number_input(k,value=float(v),key=skey,format="%.6f")
            elif isinstance(v,tuple): edited[k]=st.text_input(k,value=str(v),key=skey)
            else: edited[k]=st.text_input(k,value=str(v),key=skey)
        i+=1
    for k,v in edited.items():
        if isinstance(v,str) and v.startswith("(") and v.endswith(")"):
            try: edited[k]=eval(v)
            except Exception: pass
    return edited

with tab_single:
    chosen = st.selectbox("Choose model", config.MODEL_LIST, index=config.MODEL_LIST.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in config.MODEL_LIST else 0)
    colA, colB, colC, colD = st.columns(4)
    with colA: do_cv = st.checkbox("Cross-Validation", True)
    with colB: do_tune = st.checkbox("GridSearch", True)
    with colC: use_groups = st.checkbox("Prevent leakage (Group by 'name')", True)
    with colD: use_smote = st.checkbox("SMOTE", False)
    calibrate = st.checkbox("Calibrate (isotonic)", False)
    thr_mode = st.selectbox("Threshold strategy", ["youden","f1"], index=0)
    params = edit_params(chosen, "single_")
    if st.button("🚀 Train model", key="single_train"):
        res = mp.train_model(config.TRAIN_DATA_PATH, chosen, params, do_cv=do_cv, do_tune=do_tune, artifact_tag=f"single_{chosen}", use_groups=use_groups, use_smote=use_smote, calibrate=calibrate, thr_mode=thr_mode)
        if not res.get("ok"):
            st.error("\n".join(res.get("errors", [])))
        else:
            st.success(f"Candidate saved: {res['candidate_path']}")
            st.json(res["val_metrics"])

with tab_multi:
    pick = st.multiselect("Select models", options=config.MODEL_LIST, default=["XGBoost","RandomForest","LogisticRegression"])
    do_cv2 = st.checkbox("Cross-Validation", True, key="multi_cv")
    do_tune2 = st.checkbox("GridSearch", True, key="multi_tune")
    use_groups2 = st.checkbox("Prevent leakage (Group by 'name')", True, key="multi_groups")
    use_smote2 = st.checkbox("SMOTE", False, key="multi_smote")
    calibrate2 = st.checkbox("Calibrate", False, key="multi_calib")
    thr_mode2 = st.selectbox("Threshold", ["youden","f1"], index=0, key="multi_thr")
    param_map={}
    for m in pick:
        with st.expander(f"⚙️ Parameters — {m}", expanded=False):
            param_map[m] = edit_params(m, f"multi_{m}_")
    if st.button("🏁 Train & Compare", key="multi_train"):
        leaderboard=[]
        for m in pick:
            res = mp.train_model(config.TRAIN_DATA_PATH, m, param_map.get(m, {}), do_cv=do_cv2, do_tune=do_tune2, artifact_tag=f"multi_{m}", use_groups=use_groups2, use_smote=use_smote2, calibrate=calibrate2, thr_mode=thr_mode2)
            if res.get("ok"):
                row = res["val_metrics"].copy(); row["model_name"]=m; row["candidate_path"]=res["candidate_path"]; row["params"]=res.get("params_used", param_map.get(m, {}))
                leaderboard.append(row)
        if leaderboard:
            df_lb = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
            st.dataframe(df_lb, use_container_width=True)

with tab_best:
    st.subheader("Best Model Dashboard")
    if mp.has_production():
        meta = mp.read_best_meta()
        st.json(meta)
        try:
            ev = mp.evaluate_model(config.MODEL_PATH, artifact_tag="best_eval")
            st.json(ev["metrics"])
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("No production model yet.")

with tab_predict:
    st.subheader("Predict with the current best model")
    if not mp.has_production():
        st.warning("No production model found.")
    else:
        default_thr = 0.5
        meta = mp.read_best_meta()
        if "opt_thr" in meta: default_thr = float(meta["opt_thr"])
        thr = st.slider("Decision threshold", 0.0, 1.0, value=float(default_thr), step=0.01)
        feats = df[features]
        preds = mp.predict_with_production(feats, threshold=thr)
        st.dataframe(preds.head(20), use_container_width=True)

with tab_retrain:
    st.subheader("Retrain with additional data")
    st.caption("Upload CSV with same schema (features + target 'status').")
    up_new = st.file_uploader("Upload training CSV", type=["csv"], key="train_new")
    if st.button("🛠️ Train on uploaded data"):
        st.info("Use the full version to train & compare.")
