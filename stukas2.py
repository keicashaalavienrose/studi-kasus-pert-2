# ==========================================
# STUDI KASUS: Analisis Data & Prediksi Risiko Pembayaran
# Berdasarkan Materi Praktisi Mengajar BRIN
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# 1. Simulasi Dataset (Representasi Data PT Maju Bersama)
np.random.seed(42)
n_samples = 1000

data = {
    'customer_id': np.random.randint(101, 220, size=n_samples),
    'total_order_amount': np.random.uniform(500000, 15000000, size=n_samples),
    'item_count': np.random.randint(1, 10, size=n_samples),
    'avg_discount_given': np.random.uniform(0.0, 0.25, size=n_samples),
    'previous_delayed_count': np.random.randint(0, 5, size=n_samples),
    # Target: 1 jika pembayaran terlambat >30 hari, 0 jika lancar
    'is_late_payment': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]) 
}

df = pd.DataFrame(data)

print("--- 5 Baris Data Pertama ---")
print(df.head(), "\n")

# 2. Data Preparation & Mencegah Data Leakage
# Memisahkan Fitur (X) dan Target (y). 
# Catatan: Kolom yang menunjukkan status akhir pembayaran di masa depan harus dihindari sebagai fitur.
X = df[['total_order_amount', 'item_count', 'avg_discount_given', 'previous_delayed_count']]
y = df['is_late_payment']

# Splitting Data (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeling - Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("--- Evaluasi Logistic Regression ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred_log):.2f}")
print(classification_report(y_test, y_pred_log, zero_division=0))

# 4. Modeling - Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("--- Evaluasi Random Forest Classifier ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred_rf):.2f}")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# 5. Kesimpulan Analitis (Human-in-the-loop Interpretation)
print("\n--- Insight Analisis ---")
print("Meskipun akurasi model terlihat tinggi (~85%), perhatikan metrik Recall pada kelas minoritas (terlambat).")
print("Data yang tidak seimbang (imbalanced data) memerlukan perhatian khusus dan validasi manusia sebelum diotomatisasi ke sistem bisnis.")