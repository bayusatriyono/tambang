import numpy as np
from sklearn.linear_model import LinearRegression

# Sumber Data
X = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260, 280]).reshape(-1,1)
Y = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260, 280])

# Model Regresi
model = LinearRegression()
model.fit(X,Y)

# Koefisien
a = model.intercept_
b = model.coef_[0]

# Hasil Output Regresi
print(f"Model regresi: Y = {int(a)} + {int(b)} X")

# Hasil Prediksi penjualan
prediksi = model.predict(np.array([[300]]))
print(f"Prediksi Penjualan jika anggaran iklan Rp. 300.000 adalah : {int(prediksi[0])} unit")