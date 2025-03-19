import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os

current_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(current_folder)
neighbor_folder = os.path.join(parent_folder, "wykresy")

parent2_folder = os.path.dirname(parent_folder)
data_folder = os.path.join(parent2_folder, "dane")

# Wczytanie pliku JSONL do DataFrame
df = pd.read_json(os.path.join(data_folder, "tracks.jsonl"), lines=True)
df_session = pd.read_json(os.path.join(data_folder, "sessions.jsonl"), lines=True)
df_session = df_session[df_session["event_type"] == "play"]
# Zliczenie play_count dla każdego track_id
track_counts = (
    df_session.groupby("track_id").size().reset_index(name="service_popularity")
)
print(track_counts.head())
# Połączenie danych tracków z play_count
df = df.merge(track_counts, left_on="id", right_on="track_id", how="left")

# Wypełnienie braków (jeśli track nie miał odtworzeń)
df["service_popularity"] = df["service_popularity"].fillna(0).astype(int)

# Wyświetlenie pierwszych wierszy
df = df.drop(columns=["name", "id", "track_id"])
df["year_of_release"] = df["release_date"].str[:4].astype(int)
moved = ["service_popularity", "popularity", "id_artist", "release_date"]
pozostale_kolumny = [col for col in df.columns if col not in moved]
nowa_kolejnosc = moved + pozostale_kolumny
df = df[nowa_kolejnosc]
print(df.head())
artist_popularity = df.groupby("id_artist")["service_popularity"].sum().reset_index()

# Dodanie nazwy kolumny dla sumy
artist_popularity = artist_popularity.rename(
    columns={"service_popularity": "total_service_popularity"}
)

# artist_popularity = artist_popularity[
#     artist_popularity["total_service_popularity"] > 30
# ]
# Wyświetlenie wyniku
print(artist_popularity)
df2 = df.drop(columns=["id_artist", "release_date", "explicit", "key"])
# Rozkład sumy utworó artysty
bins = list(range(0, 200, 10))
sns.histplot(
    artist_popularity["total_service_popularity"], kde=False, color="gray", alpha=0.6
)
plt.title("Histogram sumy popularności utworów artysty")
plt.xlabel("total_service_popularity")
plt.ylabel("Liczba")
plt.savefig(os.path.join(neighbor_folder, "suma_popularności_utworów.png"))
plt.close()


# # Rozkład Year of realese

# sns.kdeplot(df["year_of_release"], fill=True, color="blue")
# plt.title("Rozkład gęstości lat wydania utworów")
# plt.xlabel("year_of_release")
# plt.savefig("histogram_lat_wydania.png")
# plt.close()

# average_popularity = df2.groupby("year_of_release")["popularity"].mean().reset_index()

# # Wykres średniej popularności od roku
# plt.figure(figsize=(10, 6))
# sns.lineplot(
#     x="year_of_release", y="popularity", data=average_popularity, marker="o", color="b"
# )

# plt.title("Średnia popularność w zależności od roku wydania", fontsize=16)
# plt.xlabel("Rok wydania", fontsize=12)
# plt.ylabel("Średnia popularność", fontsize=12)

# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(neighbor_folder, "popularnosc_a_czas.png"))
# plt.close()


# tworzenie wykresów pudełkowych

# numeric_columns = df2.select_dtypes(include=["float64", "int64"])
# num_cols = len(numeric_columns.columns)
# fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# for ax, column in zip(axes.ravel(), numeric_columns.columns):
#     ax.boxplot(
#         df[column].dropna(),
#         vert=True,
#         patch_artist=True,
#         boxprops=dict(facecolor="skyblue"),
#     )
#     ax.set_title(column, fontsize=10)
#     ax.set_xticks([])  # Usunięcie etykiet X
#     ax.set_ylabel("Wartość", fontsize=8)

# for ax in axes.ravel()[len(numeric_columns.columns) :]:
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig(os.path.join(neighbor_folder, "wykresy_pudelkowe.png"))
# plt.close()


# # Korelacja

# plt.figure(figsize=(15, 12))
# macierz_korelacji = df2.corr()
# sns.heatmap(macierz_korelacji, annot=True, cmap="coolwarm")
# plt.savefig(os.path.join(neighbor_folder, "macierz_korelacji.png"))
# plt.close()

# # # Rozkład popularity

# # sns.kdeplot(df["popularity"], fill=True, color="blue")
# # plt.title("Rozkład gęstości popularności")
# # plt.xlabel("Popularity")
# # plt.savefig(os.path.join(neighbor_folder, "histogram.png"))
# # plt.close()

# # Informacja wzajemna

# df2 = df2.fillna(df2.mean())
# # Tworzenie macierzy współczynników informacji wzajemnej
# mi_matrix = pd.DataFrame(
#     np.zeros((df2.shape[1], df2.shape[1])), columns=df2.columns, index=df2.columns
# )

# # Obliczanie informacji wzajemnej pomiędzy każdą parą kolumn
# for col1 in df2.columns:
#     for col2 in df2.columns:
#         mi_matrix.loc[col1, col2] = mutual_info_regression(df2[[col1]], df2[col2])[0]

# plt.figure(figsize=(15, 12))
# sns.heatmap(
#     mi_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, annot_kws={"size": 8}
# )
# plt.title("Macierz współczynników informacji wzajemnej", fontsize=16)
# plt.tight_layout()
# plt.savefig(os.path.join(neighbor_folder, "informacja_wzajemna.png"))

df2 = df2[df2["year_of_release"] > 2000]

# Korelacja

plt.figure(figsize=(15, 12))
macierz_korelacji = df2.corr()
sns.heatmap(macierz_korelacji, annot=True, cmap="coolwarm")
plt.savefig(os.path.join(neighbor_folder, "macierz_korelacji_po2000.png"))
plt.close()

# Rozkład popularity

sns.kdeplot(df["popularity"], fill=True, color="blue")
plt.title("Rozkład gęstości popularności")
plt.xlabel("Popularity")
plt.savefig(os.path.join(neighbor_folder, "histogram_po2000.png"))
plt.close()

# Informacja wzajemna

df2 = df2.fillna(df2.mean())
# Tworzenie macierzy współczynników informacji wzajemnej
mi_matrix = pd.DataFrame(
    np.zeros((df2.shape[1], df2.shape[1])), columns=df2.columns, index=df2.columns
)

# Obliczanie informacji wzajemnej pomiędzy każdą parą kolumn
for col1 in df2.columns:
    for col2 in df2.columns:
        mi_matrix.loc[col1, col2] = mutual_info_regression(df2[[col1]], df2[col2])[0]

plt.figure(figsize=(15, 12))
sns.heatmap(
    mi_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, annot_kws={"size": 8}
)
plt.title("Macierz współczynników informacji wzajemnej", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(neighbor_folder, "informacja_wzajemna_po2000.png"))
