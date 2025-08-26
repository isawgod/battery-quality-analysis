import os
import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Battery QC & RUL", layout="centered")
st.title("🔋 Battery Quality & RUL Estimator")

st.markdown("""
Upload cycle-level features to predict **defect probability** and estimate **Remaining Useful Life (RUL)**.

**Required columns:** `fade_rate, duration, min_voltage, mean_current, mean_temp, ambient_temp`
""")

# ---------- Safe loaders ----------
def safe_load_joblib(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
        return None
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return None

# Try local files first
clf_art = safe_load_joblib("clf_rf.joblib")
rul_art = safe_load_joblib("rul_gbr.joblib")

# ---------- Upload models if not present ----------
st.subheader("Models")
col1, col2 = st.columns(2)
with col1:
    up_clf = st.file_uploader("Upload classifier artifact (clf_rf.joblib)", type=["joblib"], key="clf_up")
    if up_clf is not None:
        try:
            clf_art = joblib.load(io.BytesIO(up_clf.read()))
            st.success("Classifier loaded from upload ✅")
        except Exception as e:
            st.error(f"Failed to load classifier: {e}")

with col2:
    up_rul = st.file_uploader("Upload RUL artifact (rul_gbr.joblib)", type=["joblib"], key="rul_up")
    if up_rul is not None:
        try:
            rul_art = joblib.load(io.BytesIO(up_rul.read()))
            st.success("RUL model loaded from upload ✅")
        except Exception as e:
            st.error(f"Failed to load RUL model: {e}")

def available(art, name):
    if art:
        st.success(f"{name} ready ✅")
    else:
        st.warning(f"{name} not found. Train & save it from the notebook (Section 11) or upload the .joblib above.")

available(clf_art, "Classifier (clf_rf.joblib)")
available(rul_art, "RUL Regressor (rul_gbr.joblib)")

st.divider()
if st.button("Download demo CSV"):
    demo = pd.DataFrame({
        "fade_rate": [-0.0018, -0.0022, -0.0040, -0.0009],
        "duration": [2900, 2850, 2400, 3050],
        "min_voltage": [2.78, 2.72, 2.60, 2.81],
        "mean_current": [1.48, 1.50, 1.55, 1.44],
        "mean_temp": [29.5, 31.2, 35.0, 28.7],
        "ambient_temp": [25.0, 25.3, 26.1, 24.8]
    })
    st.download_button("Save demo_cycles.csv", demo.to_csv(index=False), "demo_cycles.csv", use_container_width=True)

uploaded = st.file_uploader("Upload CSV with cycle features", type=["csv"])
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.write("Preview:", df_in.head())

    required = ["fade_rate","duration","min_voltage","mean_current","mean_temp","ambient_temp"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    out = df_in.copy()

    # Classification
    if clf_art:
        try:
            imp_c = clf_art["imputer"]
            mdl_c = clf_art["model"]
            feats_c = clf_art["features"]
            Xc = imp_c.transform(out[feats_c])
            out["defect_proba"] = mdl_c.predict_proba(Xc)[:, 1]
            out["defect_pred"] = (out["defect_proba"] >= 0.5).astype(int)
        except Exception as e:
            st.error(f"Classifier error: {e}")
    else:
        st.info("Classifier not available; skipping defect prediction.")

    # RUL
    if rul_art:
        try:
            imp_r = rul_art["imputer"]
            mdl_r = rul_art["model"]
            feats_r = rul_art["features"]
            Xr = imp_r.transform(out[feats_r])
            out["RUL_pred"] = mdl_r.predict(Xr)
        except Exception as e:
            st.error(f"RUL model error: {e}")
    else:
        st.info("RUL model not available; skipping RUL prediction.")

    st.subheader("Predictions")
    st.dataframe(out.head(50), use_container_width=True)
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", use_container_width=True)

st.caption("Tip: Train & export models from the notebook (Section 11) and keep artifacts <100 MB.")
