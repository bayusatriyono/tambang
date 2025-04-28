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