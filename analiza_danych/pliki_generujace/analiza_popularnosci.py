import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_selection import mutual_info_regression
import os

current_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(current_folder)
neighbor_folder = os.path.join(parent_folder, "wykresy")

parent2_folder = os.path.dirname(parent_folder)
data_folder = os.path.join(parent2_folder, "dane")

# Wczytanie pliku JSONL z danymi dotyczącymi utworów do DataFrame
df = pd.read_json(os.path.join(data_folder, "tracks.jsonl"), lines=True)

# Wczytanie pliku JSONL z danymi dotyczącymi artystów do DataFrame
df2 = pd.read_json(os.path.join(data_folder, "artists.jsonl"), lines=True)
df2 = df2.drop(columns=["name"])
df2 = df2.rename(columns={"id": "id_artist"})
df2 = df2.dropna()

# Usunięcie kolumny "name" oraz "id" z danych dotyczacych utworów
df_artist_track = df.drop(
    columns=[
        "name",
        "id",
        "duration_ms",
        "explicit",
        "release_date",
        "danceability",
        "energy",
        "key",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
)

# Usunięcie wierszy zawierających null w dowolnym polu
df_artist_track = df_artist_track.dropna()

df_artist_track_c = df_artist_track.__deepcopy__()
data = df_artist_track.merge(df2, on="id_artist")
data = data.explode("genres")  # Rozdzielenie listy gatunków

# Zakodowanie gatunków i przygotowanie danych
data["gatunki_kod"] = data["genres"].astype("category").cat.codes  # Kodowanie gatunków
X = data[["gatunki_kod"]]  # Cecha wejściowa: gatunek
y = data["popularity"]  # Cel: popularność

mi = mutual_info_regression(X, y)
print(f"Informacja wzajemna między gatunkami a popularnością: {mi[0]}")


# Przekształcenie kolumny 'id_artist' na wartości numeryczne
label_encoder = LabelEncoder()
df_artist_track["id_artist"] = label_encoder.fit_transform(df_artist_track["id_artist"])

# Obliczenie współczynnika informacji wzajemnej
mi = mutual_info_regression(
    df_artist_track[["id_artist"]], df_artist_track["popularity"]
)
print(f"Współczynnik informacji wzajemnej: {mi[0]}")

# Utworzenie nowego DataFrame z utworami, których popularność jest większa niż 75
df_artist_tracks_high_popular = df_artist_track[df_artist_track["popularity"] >= 75]

# Obliczenie współczynnika informacji wzajemnej dla nowego DataFrame
mi_high_popular = mutual_info_regression(
    df_artist_tracks_high_popular[["id_artist"]],
    df_artist_tracks_high_popular["popularity"],
)
print(f"Współczynnik informacji wzajemnej dla popularności >= 75: {mi_high_popular[0]}")


# genres = set()

# for i in range(0,len(df2) -1):
#     a = df2["genres"][i]
#     if a != None:
#         a = set(a)
#         genres = genres.union(a)

# print(genres)
