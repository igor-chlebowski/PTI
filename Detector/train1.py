#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model_hasy.py

Trenuje model CNN na zestawie HASYv2 (369 klas symboli).
Użycie:
    python train_model_hasy.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
from sklearn.model_selection import train_test_split

# ----------------------------
# 1) Ścieżki do danych
# ----------------------------
folder       = '/home/igor/Prog/opencvscanner/Model/HASY'
labels_csv   = os.path.join(folder, 'hasy-data-labels.csv')
data_dir     = os.path.join(folder, 'hasy-data')

# ----------------------------
# 2) Wczytanie etykiet
# ----------------------------
# Plik hasy-data-labels.csv zawiera nagłówek, więc header=0
df_labels = pd.read_csv(
    labels_csv,
    header=0,
    usecols=[0,1],
    names=['id','label'],
    dtype={'id':str, 'label':int},
    skiprows=1
)
# Dodaj rozszerzenie .png do nazwy pliku TYLKO jeśli nie ma go w id
df_labels['filename'] = df_labels['id'].apply(lambda x: x if x.endswith('.png') else x + '.png')

# Wczytaj symbole i zmapuj etykiety PRZED podziałem na train/test
tab = pd.read_csv(os.path.join(folder, 'symbols.csv'), header=None)
symbols = tab[0].tolist()
#symbols = [int(s) for s in symbols]  # jeśli symbole to liczby
# lub
df_labels['label'] = df_labels['label'].astype(str)
symbols = [str(s) for s in symbols]
label2idx = {label: idx for idx, label in enumerate(symbols)}
df_labels['label'] = df_labels['label'].map(label2idx)
missing = df_labels[df_labels['label'].isna()]
if not missing.empty:
    print("Brakujące etykiety (nie ma ich w symbols.csv):")
    print(missing['id'].tolist())
    print(missing['label'].tolist())
    # Usuń wiersze z NaN
    df_labels = df_labels.dropna(subset=['label'])
df_labels['label'] = df_labels['label'].astype(int)
num_classes = len(symbols)

print("Przykładowe etykiety z hasy-data-labels.csv:", df_labels['label'].unique()[:20])
print("Przykładowe symbole z symbols.csv:", symbols[:20])

# 80% dane treningowe, 20% testowe (stratyfikacja po etykietach)
train_df, test_df = train_test_split(
    df_labels,
    test_size=0.2,
    stratify=df_labels['label'],
    random_state=42
)
print(f"Dane podzielone: {len(train_df)} treningowych, {len(test_df)} testowych.")

# ----------------------------
# 3) Funkcja do ładowania obrazów
# ----------------------------
def load_images(df):
    imgs, labels = [], df['label'].values.astype('int32')
    for fname in df['filename']:
        path = os.path.join(folder, fname)  # Użyj folder zamiast data_dir
        img = Image.open(path).convert('L')    # grayscale
        arr = np.array(img).astype('float32') / 255.0
        imgs.append(arr)
    x = np.stack(imgs, axis=0)
    return x.reshape(-1, 32, 32, 1), labels

x_train, y_train = load_images(train_df)
x_test,  y_test  = load_images(test_df)

# ----------------------------
# 4) Definicja modelu CNN
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# ----------------------------
# 5) Kompilacja
# ----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 6) Trenowanie
# ----------------------------
epochs = 26
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# ----------------------------
# 7) Zapis modelu i wykres historii
# ----------------------------
output_model = 'hasy_recognition_model.h5'
model.save(output_model)
print(f"Model zapisany do: {output_model}")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('HASYv2 Training Accuracy')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.savefig('training_history_hasy.png')
print("Zapisano wykres do: training_history_hasy.png")
