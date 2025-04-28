Ini adalah tugas kelompok Mata Kuliah Data Mining Teknik Informatika UNINDRA Kelas X6I Tahun 2025.
Anggota Kelompok :

1. Bayu Satriyono - 202243500977
2. Nanda Riziah Ryan Turi - 202243501099
3. Muhamad Rizki - 202243501007
4. Ahmad Afrizal - 202243501079
5. Riska Ananda Ruslansyah - 202243501111


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv("data_layanan_masyarakat.csv", delimiter=';')
X = data[['X1', 'X2']]
y = data['Y']

# Buat model dan latih
model = LogisticRegression()
model.fit(X, y)

# Buat grid nilai untuk X1 dan X2
x1_range = np.linspace(X['X1'].min(), X['X1'].max(), 100)
x2_range = np.linspace(X['X2'].min(), X['X2'].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

# Prediksi probabilitas
probs = model.predict_proba(X_mesh)[:, 1]
probs = probs.reshape(x1_mesh.shape)

# Plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(x1_mesh, x2_mesh, probs, levels=20, cmap='RdYlBu', alpha=0.7)
plt.colorbar(contour, label="Probabilitas Kepuasan (Y=1)")
plt.scatter(X['X1'], X['X2'], c=y, cmap='bwr', edgecolor='k', s=70)
plt.xlabel("X1: Daya Tanggap")
plt.ylabel("X2: Empati")
plt.title("Pengaruh X1 dan X2 terhadap Probabilitas Kepuasan")
plt.grid(True)
plt.show()

# Import pustaka yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv(r"D:\tambang\tugas 6\data_layanan_masyarakat.csv", delimiter=';')

# 2. Pisahkan fitur (X1 dan X2) dan target (Y)
X = data[['X1', 'X2']]
y = data['Y']

# 3. Bagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Buat model regresi logistik
model = LogisticRegression()

# 5. Latih model dengan data latih
model.fit(X_train, y_train)

# 6. Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# 7. Evaluasi model
print("=== Akurasi ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))