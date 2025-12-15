import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error

# =====================
# KONFIGURASI HALAMAN
# =====================
st.set_page_config(
    page_title="Customer ML App",
    layout="centered"
)

st.title("📊 Customer Classification & Regression App")

# =====================
# LOAD DATA (AMAN)
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_customers_cleaned (1).csv")

df = load_data()

features = ["age", "income", "credit_score", "total_spent"]

# =====================
# INPUT USER
# =====================
st.header("🧑 Input Data Pelanggan")

age = st.slider("Umur", 18, 80, 30)
income = st.number_input("Pendapatan", 0, 100_000_000, 5_000_000)
credit_score = st.slider("Credit Score", 300, 850, 650)
total_spent = st.number_input("Total Pengeluaran", 0, 50_000_000, 1_000_000)

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "total_spent": total_spent
}])

# =====================
# TRAIN MODEL (DIKONTROL TOMBOL)
# =====================
st.header("⚙️ Proses Model")

if st.button("🚀 Jalankan Model"):
    with st.spinner("Melatih model, mohon tunggu..."):

        X = df[features]

        # ----- KLASIFIKASI -----
        y_class = df["subscription"]
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=30, random_state=42)
        clf.fit(Xc_train, yc_train)

        # ----- REGRESI -----
        y_reg = df["churn_risk"]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

        reg = RandomForestRegressor(n_estimators=30, random_state=42)
        reg.fit(Xr_train, yr_train)

    st.success("Model berhasil dilatih!")

    # =====================
    # HASIL PREDIKSI
    # =====================
    st.subheader("📌 Hasil Prediksi")

    status = clf.predict(input_df)[0]
    prob = clf.predict_proba(input_df)[0][1]
    churn = reg.predict(input_df)[0]

    st.write("Status Subscription:", "Berlangganan" if status == 1 else "Tidak Berlangganan")
    st.write("Probabilitas:", round(prob, 2))
    st.write("Prediksi Churn Risk:", round(churn, 2))

    # =====================
    # EVALUASI
    # =====================
    st.header("📈 Evaluasi Model")

    # Confusion Matrix
    cm = confusion_matrix(yc_test, clf.predict(Xc_test))
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.write("Accuracy:", accuracy_score(yc_test, clf.predict(Xc_test)))

    # Regresi Error
    y_pred_r = reg.predict(Xr_test)
    st.write("MAE:", mean_absolute_error(yr_test, y_pred_r))
