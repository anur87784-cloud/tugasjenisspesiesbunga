import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Perbandingan Model Klasifikasi", layout="wide")

st.title("🧠 Aplikasi Perbandingan Model Klasifikasi")
st.write("Bandingkan kinerja Logistic Regression, kNN, Random Forest, dan Decision Tree pada dataset kamu!")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("📂 Upload file CSV dataset kamu", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Awal")
    st.dataframe(data.head())

    # --- Pilih Target dan Fitur ---
    st.subheader("⚙️ Pilih Kolom Target dan Fitur")
    all_columns = data.columns.tolist()
    target_col = st.selectbox("Pilih kolom target (label)", all_columns)
    feature_cols = st.multiselect("Pilih kolom fitur (atribut)", [c for c in all_columns if c != target_col])

    if target_col and feature_cols:
        X = data[feature_cols]
        y = data[target_col]

        # --- Encode label jika perlu ---
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # --- Bagi data train dan test ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Model yang digunakan ---
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "k-Nearest Neighbors": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }

        results = {}
        st.subheader("📈 Hasil Evaluasi Model")

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

            st.write(f"### {name}")
            st.write(f"**Akurasi:** {acc:.2f}")
            st.text("Laporan Klasifikasi:")
            st.text(classification_report(y_test, y_pred))

            # --- Confusion Matrix ---
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            st.divider()

        # --- Tampilkan perbandingan akurasi ---
        st.subheader("🏆 Perbandingan Akurasi Model")
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values(by='Accuracy', ascending=False)
        st.bar_chart(result_df)

else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu untuk memulai.")
