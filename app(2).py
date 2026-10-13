import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Iris K-Means Clustering",
    page_icon="🌸",
    layout="centered"
)

MODEL_PATH = "iris_kmeans_model.pkl"

st.title("🌸 Iris K-Means Clustering")
st.write(
    "Enter the four Iris flower measurements. The app scales the inputs "
    "and predicts the cluster using your saved K-Means model."
)

@st.cache_resource
def load_saved_model(path):
    with open(path, "rb") as file:
        saved = pickle.load(file)

    # Supports either a dictionary containing model/scaler/features
    # or a pickle containing only the trained K-Means model.
    if isinstance(saved, dict):
        model = saved.get("model")
        scaler = saved.get("scaler")
        features = saved.get(
            "features",
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )
        class_names = saved.get("class_names")
        cluster_to_species = saved.get("cluster_to_species")
    else:
        model = saved
        scaler = None
        features = [
            "SepalLengthCm", "SepalWidthCm",
            "PetalLengthCm", "PetalWidthCm"
        ]
        class_names = None
        cluster_to_species = None

    if model is None:
        raise ValueError("The PKL dictionary does not contain a 'model' entry.")

    return model, scaler, features, class_names, cluster_to_species


if not os.path.exists(MODEL_PATH):
    st.error(f"Could not find `{MODEL_PATH}`.")
    st.info(
        "Place `iris_kmeans_model.pkl` in the same folder as `app.py`, "
        "then run the app again."
    )
    st.code(
        "your-project-folder/\n"
        "├── app.py\n"
        "└── iris_kmeans_model.pkl"
    )
    st.stop()

try:
    model, scaler, features, class_names, cluster_to_species = load_saved_model(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the PKL model: {error}")
    st.stop()

default_values = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2,
}

st.subheader("Flower measurements")

with st.form("iris_prediction_form"):
    values = {}
    cols = st.columns(2)

    for i, feature in enumerate(features):
        with cols[i % 2]:
            values[feature] = st.number_input(
                feature.replace("Cm", " (cm)"),
                min_value=0.0,
                max_value=10.0,
                value=float(default_values.get(feature, 1.0)),
                step=0.1,
                format="%.1f"
            )

    submitted = st.form_submit_button("Predict Cluster", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=features)

    try:
        model_input = scaler.transform(input_df) if scaler is not None else input_df
        cluster = int(model.predict(model_input)[0])

        st.success(f"Predicted cluster: {cluster}")

        # Species mapping is optional because K-Means produces cluster IDs,
        # not species names, unless a mapping was saved with the model.
        if cluster_to_species is not None:
            species = cluster_to_species.get(
                cluster, cluster_to_species.get(str(cluster))
            )
            if species is not None:
                st.metric("Mapped species", str(species))
        else:
            st.caption(
                "K-Means returns a cluster number, not a guaranteed species name. "
                "To display species names, save a cluster-to-species mapping "
                "along with the model."
            )

        st.subheader("Input used")
        st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.warning(
            "Check that the PKL file was created using the same feature order "
            "and preprocessing steps as this app."
        )

st.divider()
st.caption("Model: K-Means clustering | Dataset: Iris")
