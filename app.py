import io
import os
import joblib
import pandas as pd
import streamlit as st

logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NED_Logo_JPG.jpg")

st.set_page_config(
    page_title="Bond Strength of FRP Bars Prediction Tool",
    page_icon=logo_path if os.path.exists(logo_path) else None,
    layout="wide",
)

@st.cache_resource
def load_bundle():
    bundle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.joblib")
    return joblib.load(bundle_file)

bundle = load_bundle()
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
target_columns = bundle["target_columns"]
feature_schema = bundle["feature_schema"]
hyperparameters = bundle["hyperparameters"]
metrics = bundle["metrics"]
n_samples = bundle["n_samples"]

# Header with Logo
col_logo, col_title = st.columns([1, 6], vertical_alignment="center")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=110)

with col_title:
    st.title("Bond Strength of FRP Bars Prediction Tool")
    st.markdown(
        "**Developed by Muhammad Saad Ifrahim**, Department of Civil Engineering, "
        "NED University of Engineering and Technology."
    )
    st.caption(
        f"SMA-XGBoost Model | Trained on {n_samples} experimental records | Targets: {', '.join(target_columns)}"
    )

st.divider()

# Sidebar: Model Performance & Hyperparameters
with st.sidebar:
    st.header("Model Summary")
    for target, m in metrics.items():
        st.subheader(f"Validation ({target})")
        col_a, col_b = st.columns(2)
        col_a.metric("Test R²", f"{m['Test R2']:.4f}")
        col_b.metric("Test RMSE", f"{m['Test RMSE']:.3f}")
        col_c, col_d = st.columns(2)
        col_c.metric("Test MAE", f"{m['Test MAE']:.3f}")
        col_d.metric("Test a20", f"{m['Test a20-index (%)']:.2f}%")
        st.divider()

    with st.expander("Optimized Hyperparameters"):
        hp_df = pd.DataFrame(
            list(hyperparameters.items()), columns=["Parameter", "Value"]
        )
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "Single Prediction",
    "Batch Prediction (Excel / CSV)",
    "Dataset Input Ranges"
])

# ------------------------------------------
# TAB 1: SINGLE PREDICTION
# ------------------------------------------
with tab1:
    st.subheader("Enter Input Parameters")

    with st.form("prediction_form"):
        cols = st.columns(2)
        user_inputs = {}

        for idx, col_name in enumerate(feature_columns):
            meta = feature_schema[col_name]
            with cols[idx % 2]:
                if meta["type"] == "categorical":
                    user_inputs[col_name] = st.selectbox(
                        label=col_name,
                        options=meta["categories"],
                    )
                else:
                    min_v = meta["min"]
                    max_v = meta["max"]
                    med_v = meta["median"]
                    help_text = f"Training range: {min_v:.3g} to {max_v:.3g} (Median: {med_v:.3g})"

                    if meta["is_int"]:
                        user_inputs[col_name] = st.number_input(
                            label=col_name,
                            value=int(round(med_v)),
                            step=1,
                            help=help_text,
                        )
                    else:
                        step_size = max(round((max_v - min_v) / 100.0, 4), 0.01)
                        user_inputs[col_name] = st.number_input(
                            label=col_name,
                            value=float(round(med_v, 4)),
                            step=step_size,
                            format="%.4f",
                            help=help_text,
                        )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_inputs])[feature_columns]
        preds = pipeline.predict(input_df)

        st.success("Prediction Complete")
        out_cols = st.columns(len(target_columns))
        for i, target in enumerate(target_columns):
            out_cols[i].metric(
                label=f"Predicted {target} (MPa)",
                value=f"{preds[0, i]:.4f}",
            )

        out_of_range = []
        for col_name in feature_columns:
            meta = feature_schema[col_name]
            if meta["type"] == "numeric":
                val = user_inputs[col_name]
                if val < meta["min"] or val > meta["max"]:
                    out_of_range.append(
                        f"{col_name} ({val} outside [{meta['min']:.3g}, {meta['max']:.3g}])"
                    )
        if out_of_range:
            st.warning(
                "Extrapolation warning: The following inputs fall outside the training data range: "
                + "; ".join(out_of_range)
            )

# ------------------------------------------
# TAB 2: BATCH PREDICTION
# ------------------------------------------
with tab2:
    st.subheader("Upload Excel or CSV for Batch Predictions")
    st.write(f"Required columns: `{', '.join(feature_columns)}`")

    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df_batch = pd.read_csv(uploaded_file)
        else:
            df_batch = pd.read_excel(uploaded_file)

        missing = [c for c in feature_columns if c not in df_batch.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X_batch = df_batch[feature_columns].copy()
            for col_name in feature_columns:
                if feature_schema[col_name]["type"] == "categorical":
                    X_batch[col_name] = X_batch[col_name].astype(str).str.strip()

            batch_preds = pipeline.predict(X_batch)
            df_out = df_batch.copy()
            for i, target in enumerate(target_columns):
                df_out[f"Predicted_{target}"] = batch_preds[:, i]

            st.dataframe(df_out, use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Predictions")

            st.download_button(
                label="Download Predictions (.xlsx)",
                data=buffer.getvalue(),
                file_name="frp_bond_strength_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

# ------------------------------------------
# TAB 3: DATASET INPUT RANGES
# ------------------------------------------
with tab3:
    st.subheader("Feature Summary from Training Database")
    rows = []
    for col_name in feature_columns:
        meta = feature_schema[col_name]
        if meta["type"] == "numeric":
            rows.append({
                "Feature": col_name,
                "Type": "Numeric",
                "Min": round(meta["min"], 4),
                "Median": round(meta["median"], 4),
                "Max": round(meta["max"], 4),
                "Categories": "-",
            })
        else:
            rows.append({
                "Feature": col_name,
                "Type": "Categorical",
                "Min": "-",
                "Median": "-",
                "Max": "-",
                "Categories": ", ".join(meta["categories"]),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
