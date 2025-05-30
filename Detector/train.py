#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py

Skrypt do trenowania modelu rozpoznającego cyfry 0–9.
Dane MNIST w formacie .npy powinny być w folderze:
    /home/igor/Prog/opencvscanner/Model/minist

Uruchomienie:
    python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# 1) Ścieżka do folderu z danymi
# ----------------------------
folder = '/home/igor/Prog/opencvscanner/Model/minist'

# ----------------------------
# 2) Wczytanie danych
# ----------------------------
x_train = np.load(os.path.join(folder, 'x_train.npy'))
y_train = np.load(os.path.join(folder, 'y_train.npy'))
x_test  = np.load(os.path.join(folder, 'x_test.npy'))
y_test  = np.load(os.path.join(folder, 'y_test.npy'))

# ----------------------------
# 3) Preprocessing
# ----------------------------
# – dodanie kanału (1), konwersja do float32 i normalizacja [0,1]
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# ----------------------------
# 4) Definicja modelu
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
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
epochs = 5
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# ----------------------------
# 7) Zapis modelu
# ----------------------------
output_path = 'digit_recognition_model.h5'
model.save(output_path)
print(f"Model wytrenowany i zapisany do: {output_path}")

# ----------------------------
# (Opcjonalnie) Zapis wykresu historii
# ----------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Dokładność podczas trenowania')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.savefig('training_history.png')
print("Zapisano wykres historii jako training_history.png")
