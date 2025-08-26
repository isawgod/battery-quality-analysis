import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Battery QC & RUL", layout="centered")
st.title("🔋 Battery Quality & RUL Estimator")

# Load artifacts
clf_art = joblib.load("clf_rf.joblib")
rul_art = joblib.load("rul_gbr.joblib")

clf_imp, clf_model, clf_feats = clf_art["imputer"], clf_art["model"], clf_art["features"]
rul_imp, rul_model, rul_feats = rul_art["imputer"], rul_art["model"], rul_art["features"]

st.markdown("Upload cycle-level feature CSV with columns:")
st.code(", ".join(sorted(set(clf_feats) | set(rul_feats))))

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    df_in = pd.read_csv(file)

    missing_clf = [c for c in clf_feats if c not in df_in.columns]
    missing_rul = [c for c in rul_feats if c not in df_in.columns]
    if missing_clf or missing_rul:
        st.error(f"Missing columns: {set(missing_clf) | set(missing_rul)}")
    else:
        Xc = clf_imp.transform(df_in[clf_feats])
        yc_proba = clf_model.predict_proba(Xc)[:, 1]
        yc_pred = (yc_proba >= 0.5).astype(int)

        Xr = rul_imp.transform(df_in[rul_feats])
        yr_pred = rul_model.predict(Xr)

        st.subheader("Predictions")
        out = df_in.copy()
        out["defect_proba"] = yc_proba
        out["defect_pred"] = yc_pred
        out["RUL_pred"] = yr_pred
        st.dataframe(out.head(30))
        st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv")
