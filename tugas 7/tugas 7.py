import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math

# === BACA DATASET ===
df = pd.read_csv(r"D:\tambang\tugas 7\data_perumahan.csv")  # Pastikan file ada di direktori yang sama

# === ENKODE SETIAP KOLOM KATEGORIKAL ===
encoders = {col: LabelEncoder() for col in df.columns}
for col in df.columns:
    df[col] = encoders[col].fit_transform(df[col])

# === PISAHKAN FITUR DAN TARGET ===
X = df[['C1', 'C2', 'C3']]
y = df['C4']

# === LATIH MODEL ===
model = CategoricalNB()
model.fit(X, y)

# === DATA BARU YANG INGIN DIKLASIFIKASIKAN ===
input_data = {
    "C1": "Sedang",
    "C2": "Dekat",
    "C3": "Ya"  # Pastikan ini salah satu label yang ada dalam data pelatihan
}

# === ENKODE INPUT DATA ===
input_encoded = []
for feature in ['C1', 'C2', 'C3']:
    if input_data[feature] in encoders[feature].classes_:
        encoded_value = encoders[feature].transform([input_data[feature]])[0]
        input_encoded.append(encoded_value)
    else:
        print(f"Error: Nilai {input_data[feature]} tidak ditemukan pada kolom {feature}.")
        input_encoded.append(-1)  # Gantilah dengan nilai yang sesuai, seperti -1 jika tidak valid

# === PREDIKSI ===
input_encoded_df = pd.DataFrame([input_encoded], columns=X.columns)  # Menyusun input sebagai DataFrame
prediction_encoded = model.predict(input_encoded_df)[0]
prediction = encoders['C4'].inverse_transform([prediction_encoded])[0]

# === CETAK HASIL ===
print("=== DATA INPUT ===")
for key, value in input_data.items():
    print(f" - {key}: {value}")

# === PERHITUNGAN POSTERIOR PROBABILITIES SECARA MANUAL ===
print("\n=== PERHITUNGAN POSTERIOR PROBABILITIES ===")

class_log_prior = model.class_log_prior_  # log(P(class))
feature_log_prob = model.feature_log_prob_  # log(P(feature | class))

# Menampilkan informasi kelas dan log prior
classes = model.classes_
for idx, cls in enumerate(classes):
    print(f"\nKelas: {encoders['C4'].inverse_transform([cls])[0]}")
    total_log_prob = class_log_prior[idx]
    print(f"  log(P({encoders['C4'].inverse_transform([cls])[0]})) = {total_log_prob:.4f}")

    # Proses setiap fitur dan hitung log-probabilitas secara manual
    for i, feature_value in enumerate(input_encoded):
        # Mengambil log-probabilitas dari feature_log_prob
        log_prob_value = feature_log_prob[i][cls][feature_value]
        prob_value = math.exp(log_prob_value)

        print(f"  P({X.columns[i]}={input_data[X.columns[i]]} | {encoders['C4'].inverse_transform([cls])[0]}) = {prob_value:.4f} (log = {log_prob_value:.4f})")
        
        # Menambahkan log-probabilitas fitur ke total log-probabilitas
        total_log_prob += log_prob_value
    
    # Total log-posterior probability untuk kelas
    print(f"  Total log posterior: {total_log_prob:.4f}")

print(f"\n=== HASIL KLASIFIKASI ===")
print(f"Lokasi ini {'dipilih' if prediction == 'Ya' else 'tidak dipilih'} untuk perumahan ({prediction})")
