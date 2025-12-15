import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Customer ML App", layout="centered")
st.title("📊 Customer Classification & Regression App")

# Load data
df = pd.read_csv("synthetic_customers_cleaned.csv")

features = ["age", "income", "credit_score", "total_spent"]
X = df[features]

# =====================
# KLASIFIKASI
# =====================
st.header("🔵 Klasifikasi: Subscription")

y_class = df["subscription"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# =====================
# REGRESI
# =====================
st.header("🟢 Regresi: Churn Risk")

y_reg = df["churn_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred_reg)
ax2.set_xlabel("Actual Churn Risk")
ax2.set_ylabel("Predicted Churn Risk")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)
