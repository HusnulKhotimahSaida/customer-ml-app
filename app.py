import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_absolute_error
)

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Customer Classification & Regression App",
    layout="centered"
)

st.title("📊 Customer Classification & Regression App")
st.markdown(
    "Aplikasi ini menggunakan **Ensemble Method (Random Forest)** "
    "untuk melakukan **klasifikasi subscription** dan **regresi churn risk** pelanggan."
)

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_customers_cleaned (1).csv")

df = load_data()

features = ["age", "income", "credit_score", "total_spent"]
X = df[features]

# =========================
# TRAIN MODEL (CACHED)
# =========================
@st.cache_resource
def train_models(df):
    X = df[features]

    # ----- KLASIFIKASI -----
    y_class = df["subscription"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )
    clf.fit(Xc_train, yc_train)

    # ----- REGRESI -----
    y_reg = df["churn_risk"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )
    reg.fit(Xr_train, yr_train)

    return clf, reg, Xc_test, yc_test, Xr_test, yr_test


clf, reg, Xc_test, yc_test, Xr_test, yr_test = train_models(df)

# =========================
# INPUT USER
# =========================
st.header("🧑 Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umur", 18, 80, 30)
    income = st.number_input(
        "Pendapatan",
        min_value=0,
        value=5000000,
        step=500000
    )

with col2:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    total_spent = st.number_input(
        "Total Pengeluaran",
        min_value=0,
        value=1000000,
        step=100000
    )

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "total_spent": total_spent
}])

# =========================
# PREDIKSI USER
# =========================
st.subheader("📌 Hasil Prediksi")

# Klasifikasi
pred_class = clf.predict(input_df)[0]
pred_proba = clf.predict_proba(input_df)[0][1]

if pred_class == 1:
    st.success(
        f"Status Subscription: **BERLANGGANAN**  \n"
        f"Probabilitas: **{pred_proba:.2f}**"
    )
else:
    st.warning(
        f"Status Subscription: **TIDAK BERLANGGANAN**  \n"
        f"Probabilitas: **{pred_proba:.2f}**"
    )

# Regresi
pred_churn = reg.predict(input_df)[0]
st.info(f"Prediksi Churn Risk: **{pred_churn:.2f}**")

# =========================
# EVALUASI & VISUALISASI
# =========================
st.header("📈 Evaluasi Model")

# ----- Confusion Matrix -----
y_pred_c = clf.predict(Xc_test)
cm = confusion_matrix(yc_test, y_pred_c)

fig1, ax1 = plt.subplots()
ax1.imshow(cm)
ax1.set_title("Confusion Matrix - Subscription")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

st.write("Accuracy:", accuracy_score(yc_test, y_pred_c))

# ----- Regresi Plot -----
y_pred_r = reg.predict(Xr_test)

fig2, ax2 = plt.subplots()
ax2.scatter(Xr_test.index, y_pred_r, label="Predicted", alpha=0.6)
ax2.scatter(Xr_test.index, yr_test, label="Actual", alpha=0.6)
ax2.set_title("Actual vs Predicted Churn Risk")
ax2.set_xlabel("Data Index")
ax2.set_ylabel("Churn Risk")
ax2.legend()
st.pyplot(fig2)

st.write("MAE:", mean_absolute_error(yr_test, y_pred_r))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "© Customer ML App | Ensemble Method (Random Forest) | Streamlit Deployment"
)
