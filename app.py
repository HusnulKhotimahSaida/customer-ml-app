import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error

st.set_page_config(page_title="Customer Classification & Regression App", layout="centered")
st.title("📊 Customer Classification & Regression App")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("synthetic_customers_cleaned (1).csv")

features = ["age", "income", "credit_score", "total_spent"]
X = df[features]

# =====================
# TRAIN MODEL
# =====================
# Klasifikasi
y_class = df["subscription"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)

# Regresi
y_reg = df["churn_risk"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)

# =====================
# INPUT USER
# =====================
st.header("🧑 Input Data Pelanggan")

age = st.slider("Umur", 18, 80, 30)
income = st.number_input("Pendapatan", min_value=0, value=5000000)
credit_score = st.slider("Credit Score", 300, 850, 650)
total_spent = st.number_input("Total Pengeluaran", min_value=0, value=1000000)

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "total_spent": total_spent
}])

# =====================
# PREDIKSI USER
# =====================
st.subheader("📌 Hasil Prediksi")

# Klasifikasi
pred_class = clf.predict(input_df)[0]
pred_proba = clf.predict_proba(input_df)[0][1]

if pred_class == 1:
    st.success(f"Status Subscription: BERLANGGANAN (Probabilitas: {pred_proba:.2f})")
else:
    st.warning(f"Status Subscription: TIDAK BERLANGGANAN (Probabilitas: {pred_proba:.2f})")

# Regresi
pred_churn = reg.predict(input_df)[0]
st.info(f"Prediksi Churn Risk: {pred_churn:.2f}")

# =====================
# VISUALISASI EVALUASI
# =====================
st.header("📈 Evaluasi Model")

# Confusion Matrix
y_pred_c = clf.predict(X_test_c)
cm = confusion_matrix(y_test_c, y_pred_c)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix - Subscription")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.write("Accuracy:", accuracy_score(y_test_c, y_pred_c))

# Regresi Plot
y_pred_r = reg.predict(X_test_r)

fig2, ax2 = plt.subplots()
ax2.scatter(y_test_r, y_pred_r)
ax2.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()],
         linestyle="--")
ax2.set_xlabel("Actual Churn Risk")
ax2.set_ylabel("Predicted Churn Risk")
ax2.set_title("Actual vs Predicted Churn Risk")
st.pyplot(fig2)

st.write("MAE:", mean_absolute_error(y_test_r, y_pred_r))
