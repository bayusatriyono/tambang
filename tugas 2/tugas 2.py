import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file tv.csv
data = pd.read_csv(r'D:\tambang\tugas 2\tv.csv')

# membuat scatter plot dengan lib pandas
data.plot.scatter(x='av_rating', y='seasonNumber', alpha=0.5)

# menampilkan plot
plt.title("Hubungan antara Peringkat rata-rata dan jumlah musim")
plt.xlabel("Peringkat rata-rata")
plt.ylabel("jumlah musim")
plt.show()