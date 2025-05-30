import os
import pandas as pd

folder     = '/home/igor/Prog/opencvscanner/Model/HASY'
labels_csv = os.path.join(folder, 'hasy-data-labels.csv')
data_dir   = os.path.join(folder, 'hasy-data')

# sprawdź, czy ścieżka jest OK:
print(labels_csv, "exists?", os.path.exists(labels_csv))

# wczytujemy wszystkie etykiety i dzielimy 80/20
labels_df = pd.read_csv(labels_csv, header=None, names=['id','label'])
labels_df['filename'] = labels_df['id'].astype(str) + '.png'

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df['label'],
    random_state=42
)