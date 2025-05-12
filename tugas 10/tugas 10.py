import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Data dari CSV, mulai dari baris kedua dan ambil kolom: Kelas, Kulit, Warna, Ukuran, Bau
data = pd.read_csv(r"D:\tambang\tugas 10\data_buah.csv")
data = data.iloc[:, 1:6]
data.columns = ['Kelas', 'Kulit', 'Warna', 'Ukuran', 'Bau']

# Encode data kategorikal
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])

# Split fitur dan target
X = data.drop('Kelas', axis=1)
y = data['Kelas']

# Buat decision tree
tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
tree.fit(X, y)

# Visualisasi pohon
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=['Aman', 'Berbahaya'], filled=True)
plt.show()
