import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv(r'D:\tambang\tugas 3\train.csv')
# Mengisi missing values
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']]).ravel()

df['Cabin'] = df['Cabin'].fillna('Unknown')  # Perbaikan inplace=True

# Konversi kategori ke numerik
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Normalisasi 'Age' dan 'Fare'
df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

# Hapus kolom yang tidak diperlukan
df.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)

# Tampilkan hasil akhir
print(df.head())