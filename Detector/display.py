import numpy as np
import matplotlib.pyplot as plt

# 1. Wczytaj dane]
folder = '/home/igor/Prog/opencvscanner/Model/minist'

x_test = np.load(folder + '/x_test.npy')

# sprawdź pierwsze 10 etykiet



# 2. Wybierz indeks obrazka (np. 0)
idx = 3
img = x_test[idx]

# 3. Wyświetl
plt.imshow(img, cmap='gray')
plt.title(f'Cyfra: {x_test[idx]}')
plt.axis('off')
plt.show()