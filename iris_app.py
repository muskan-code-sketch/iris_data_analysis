import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classifier")
st.markdown("Train a Random Forest model on the Iris dataset and predict species.")

IRIS_DATA = """5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.2,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.6,1.4,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3.0,5.8,2.2,Iris-virginica
7.6,3.0,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3.0,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6.0,2.2,5.0,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3.0,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2.0,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3.0,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6.0,3.0,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3.0,5.2,2.3,Iris-virginica
6.3,2.5,5.0,1.9,Iris-virginica
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica"""

COLS = ["sepallength", "sepalwidth", "petallength", "petalwidth", "class"]
SPECIES_EMOJI = {
    "Iris-setosa": "🌼",
    "Iris-versicolor": "🌺",
    "Iris-virginica": "🌸"
}

@st.cache_resource
def train_model():
    df = pd.read_csv(StringIO(IRIS_DATA), header=None, names=COLS)
    # Use .to_numpy() to avoid PyArrow/pandas indexing issues on Python 3.14
    X = df[["sepallength","sepalwidth","petallength","petalwidth"]].to_numpy(dtype=float)
    y = df["class"].to_numpy(dtype=str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return clf, acc, report, df

model, accuracy, report, df = train_model()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explore Data", "📈 Model Performance"])

# ── TAB 1: PREDICT ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict Iris Species")
    st.markdown("Adjust the sliders to input flower measurements:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.5, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.5, 0.1)

    if st.button("🔍 Predict Species", use_container_width=True):
        input_arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
        prediction = model.predict(input_arr)[0]
        probabilities = model.predict_proba(input_arr)[0]
        classes = model.classes_

        emoji = SPECIES_EMOJI.get(prediction, "🌿")
        st.success(f"### {emoji} Predicted Species: **{prediction}**")
        st.markdown("**Confidence Scores:**")
        for cls, prob in zip(classes, probabilities):
            st.progress(float(prob), text=f"{SPECIES_EMOJI.get(cls,'')} {cls}: {prob*100:.1f}%")

    st.markdown("---")
    st.markdown("**Or upload a CSV** (columns: sepallength, sepalwidth, petallength, petalwidth) for batch predictions:")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            X_user = user_df.to_numpy(dtype=float)
            preds = model.predict(X_user)
            user_df["Predicted Species"] = preds
            st.dataframe(user_df, use_container_width=True)
            csv_out = io.StringIO()
            user_df.to_csv(csv_out, index=False)
            st.download_button("⬇️ Download Predictions", csv_out.getvalue(), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ── TAB 2: EXPLORE DATA ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", 4)
    col3.metric("Species", df["class"].nunique())

    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Average Feature Value by Species")
    feature = st.selectbox("Select Feature", ["sepallength","sepalwidth","petallength","petalwidth"])
    chart_data = (
        df.groupby("class")[feature]
        .mean()
        .reset_index()
        .rename(columns={"class": "Species", feature: f"Avg {feature}"})
        .set_index("Species")
    )
    st.bar_chart(chart_data)

    st.subheader("Species Count")
    st.bar_chart(df["class"].value_counts())

# ── TAB 3: MODEL PERFORMANCE ────────────────────────────────────────────────
with tab3:
    st.subheader("Model: Random Forest Classifier")
    st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")

    st.markdown("**Per-class Performance:**")
    perf_rows = []
    for species in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]:
        if species in report:
            r = report[species]
            perf_rows.append({
                "Species": f"{SPECIES_EMOJI.get(species,'')} {species}",
                "Precision": f"{r['precision']:.2f}",
                "Recall":    f"{r['recall']:.2f}",
                "F1-Score":  f"{r['f1-score']:.2f}",
                "Support":   int(r["support"])
            })
    st.table(pd.DataFrame(perf_rows))

    st.markdown("**Feature Importances:**")
    feat_imp = pd.Series(
        model.feature_importances_,
        index=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    )
    st.bar_chart(feat_imp.sort_values(ascending=False))
